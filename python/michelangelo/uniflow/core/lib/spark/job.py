"""Python implementation of spark.create_job / spark.sensor_job builtins.

Mirrors the Go/Starlark spark builtins (create_job, sensor_job) as pure-Python
functions using APIClient.SparkJobService, following the same pattern as
plugins/pipeline/run.py (create_pipeline_run / poll_pipeline_run / run_pipeline).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import grpc

from michelangelo.api.v2 import APIClient
from michelangelo.gen.api.conditions_pb2 import (
    CONDITION_STATUS_FALSE,
    CONDITION_STATUS_TRUE,
)
from michelangelo.gen.api.v2.pod_pb2 import PodSpec, ResourceSpec
from michelangelo.gen.api.v2.spark_job_pb2 import (
    Dependencies,
    DriverSpec,
    ExecutorSpec,
    SparkJob,
    SparkJobSpec,
)
from michelangelo.gen.k8s.io.apimachinery.pkg.apis.meta.v1.generated_pb2 import (
    CreateOptions,
    GetOptions,
    ObjectMeta,
)
from michelangelo.uniflow.core import star_plugin

log = logging.getLogger(__name__)

SUCCEEDED_CONDITION_TYPE = "Succeeded"
KILLED_CONDITION_TYPE = "Killed"

_DEFAULT_TIMEOUT_SECONDS = 10 * 365 * 24 * 60 * 60
_DEFAULT_POLL_SECONDS = 10


def _spark_job_to_dict(job: SparkJob) -> dict[str, Any]:
    conditions = []
    for c in job.status.status_conditions:
        conditions.append(
            {
                "type": c.type,
                "status": c.status,
                "reason": c.reason,
                "message": c.message,
            }
        )

    return {
        "metadata": {
            "name": job.metadata.name,
            "namespace": job.metadata.namespace,
        },
        "status": {
            "status_conditions": conditions,
            "job_url": job.status.job_url,
            "application_id": job.status.application_id,
        },
    }


def create_spark_job(
    namespace: str,
    main_application_file: str,
    main_class: str | None = None,
    args: list[str] | None = None,
    driver_cpu: int | None = None,
    driver_memory: str | None = None,
    executor_cpu: int | None = None,
    executor_memory: str | None = None,
    executor_instances: int | None = None,
    spark_conf: dict[str, str] | None = None,
    deps_jars: list[str] | None = None,
    deps_py_files: list[str] | None = None,
    spark_version: str = "3.5.5",
) -> SparkJob:
    """Build a SparkJob proto and submit it via the API.

    Returns the created SparkJob as returned by the server.
    """
    driver = DriverSpec()
    driver_resource = ResourceSpec()
    if driver_cpu is not None:
        driver_resource.cpu = driver_cpu
    if driver_memory is not None:
        driver_resource.memory = driver_memory
    driver_pod = PodSpec(resource=driver_resource)
    driver.pod.CopyFrom(driver_pod)

    executor = ExecutorSpec()
    executor_resource = ResourceSpec()
    if executor_cpu is not None:
        executor_resource.cpu = executor_cpu
    if executor_memory is not None:
        executor_resource.memory = executor_memory
    executor_pod = PodSpec(resource=executor_resource)
    executor.pod.CopyFrom(executor_pod)
    if executor_instances is not None:
        executor.instances = executor_instances

    deps = Dependencies()
    if deps_jars:
        deps.jars.extend(deps_jars)
    if deps_py_files:
        deps.py_files.extend(deps_py_files)

    spec = SparkJobSpec(
        main_application_file=main_application_file,
        spark_version=spark_version,
    )
    if main_class:
        spec.main_class = main_class
    if args:
        spec.main_args.extend(args)
    if spark_conf:
        for k, v in spark_conf.items():
            spec.spark_conf[k] = v
    spec.driver.CopyFrom(driver)
    spec.executor.CopyFrom(executor)
    spec.deps.CopyFrom(deps)

    spark_job = SparkJob()
    spark_job.metadata.CopyFrom(
        ObjectMeta(namespace=namespace, generateName="uniflow-splg-")
    )
    spark_job.spec.CopyFrom(spec)

    log.info(
        "Creating spark job in namespace %s with entrypoint %s",
        namespace,
        main_application_file,
    )
    created = APIClient.SparkJobService.create_spark_job(
        spark_job=spark_job, create_options=CreateOptions()
    )
    return created


def poll_spark_job(
    namespace: str,
    name: str,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    poll_seconds: int = _DEFAULT_POLL_SECONDS,
) -> dict[str, Any]:
    """Poll a SparkJob until it reaches a terminal state or times out.

    Returns a dict with metadata and status (matching run_pipeline's return
    convention). Raises RuntimeError on failure/killed, TimeoutError on timeout.
    """
    log.info(
        "Monitoring spark job %s in namespace %s (timeout=%ds, poll_interval=%ds)",
        name,
        namespace,
        timeout_seconds,
        poll_seconds,
    )

    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            current_job = APIClient.SparkJobService.get_spark_job(
                namespace=namespace, name=name, get_options=GetOptions()
            )

            for cond in current_job.status.status_conditions:
                if (
                    cond.type == KILLED_CONDITION_TYPE
                    and cond.status == CONDITION_STATUS_TRUE
                ):
                    raise RuntimeError(f"Spark job {name} was killed: {cond.message}")

                if (
                    cond.type == SUCCEEDED_CONDITION_TYPE
                    and cond.status == CONDITION_STATUS_TRUE
                ):
                    log.info("Spark job %s succeeded", name)
                    return _spark_job_to_dict(current_job)

                if (
                    cond.type == SUCCEEDED_CONDITION_TYPE
                    and cond.status == CONDITION_STATUS_FALSE
                ):
                    raise RuntimeError(f"Spark job {name} failed: {cond.message}")

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise RuntimeError(
                    f"Spark job {name} not found in namespace {namespace}"
                ) from e
            elif e.code() in (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
            ):
                log.debug("Transient error polling spark job %s: %s", name, e.details())
            elif e.code() in (
                grpc.StatusCode.PERMISSION_DENIED,
                grpc.StatusCode.UNAUTHENTICATED,
                grpc.StatusCode.INVALID_ARGUMENT,
            ):
                raise RuntimeError(
                    f"Failed to get spark job {name}: {e.details()}"
                ) from e
            else:
                log.debug("Error polling spark job %s: %s", name, e.details())
        except RuntimeError:
            raise
        except Exception as e:
            log.debug("Error polling spark job %s: %s", name, e)

        time.sleep(poll_seconds)

    raise TimeoutError(f"Spark job {name} timed out after {timeout_seconds} seconds")


@star_plugin("spark.create_job")
def create_job(
    namespace: str,
    main_application_file: str,
    main_class: str | None = None,
    args: list[str] | None = None,
    driver_cpu: int | None = None,
    driver_memory: str | None = None,
    executor_cpu: int | None = None,
    executor_memory: str | None = None,
    executor_instances: int | None = None,
    spark_conf: dict[str, str] | None = None,
    deps_jars: list[str] | None = None,
    deps_py_files: list[str] | None = None,
    spark_version: str = "3.5.5",
) -> dict[str, Any]:
    """Submit a SparkJob and return its metadata/status as a dict."""
    created = create_spark_job(
        namespace=namespace,
        main_application_file=main_application_file,
        main_class=main_class,
        args=args,
        driver_cpu=driver_cpu,
        driver_memory=driver_memory,
        executor_cpu=executor_cpu,
        executor_memory=executor_memory,
        executor_instances=executor_instances,
        spark_conf=spark_conf,
        deps_jars=deps_jars,
        deps_py_files=deps_py_files,
        spark_version=spark_version,
    )
    return _spark_job_to_dict(created)


@star_plugin("spark.sensor_job")
def sensor_job(
    namespace: str,
    name: str,
    timeout_seconds: int = 0,
    poll_seconds: int = _DEFAULT_POLL_SECONDS,
) -> dict[str, Any]:
    """Poll a SparkJob until terminal state. Returns metadata/status dict."""
    if timeout_seconds == 0:
        timeout_seconds = _DEFAULT_TIMEOUT_SECONDS

    return poll_spark_job(
        namespace=namespace,
        name=name,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
