"""Tests for michelangelo.uniflow.core.lib.spark.job builtins."""

from __future__ import annotations

from unittest.mock import patch

import grpc
import pytest

from michelangelo.gen.api.conditions_pb2 import (
    CONDITION_STATUS_FALSE,
    CONDITION_STATUS_TRUE,
    Condition,
)
from michelangelo.gen.api.v2.spark_job_pb2 import (
    SparkJob,
    SparkJobSpec,
)
from michelangelo.gen.k8s.io.apimachinery.pkg.apis.meta.v1.generated_pb2 import (
    ObjectMeta,
)
from michelangelo.uniflow.core.lib.spark.job import (
    KILLED_CONDITION_TYPE,
    SUCCEEDED_CONDITION_TYPE,
    _spark_job_to_dict,
    create_job,
    create_spark_job,
    poll_spark_job,
    sensor_job,
)


def _make_spark_job(
    name: str = "test-job",
    namespace: str = "test-ns",
    conditions: list[Condition] | None = None,
) -> SparkJob:
    job = SparkJob()
    job.metadata.CopyFrom(ObjectMeta(name=name, namespace=namespace))
    job.spec.CopyFrom(
        SparkJobSpec(
            main_application_file="s3://bucket/app.jar",
            main_class="com.example.Main",
        )
    )
    if conditions:
        for c in conditions:
            job.status.status_conditions.append(c)
    return job


def _succeeded_condition() -> Condition:
    return Condition(
        type=SUCCEEDED_CONDITION_TYPE,
        status=CONDITION_STATUS_TRUE,
        message="Job completed successfully",
    )


def _killed_condition() -> Condition:
    return Condition(
        type=KILLED_CONDITION_TYPE,
        status=CONDITION_STATUS_TRUE,
        message="Job was killed",
    )


def _failed_condition() -> Condition:
    return Condition(
        type=SUCCEEDED_CONDITION_TYPE,
        status=CONDITION_STATUS_FALSE,
        message="Job failed",
    )


class TestSparkJobToDict:
    def test_basic_conversion(self):
        job = _make_spark_job(conditions=[_succeeded_condition()])
        result = _spark_job_to_dict(job)

        assert result["metadata"]["name"] == "test-job"
        assert result["metadata"]["namespace"] == "test-ns"
        assert len(result["status"]["status_conditions"]) == 1
        assert (
            result["status"]["status_conditions"][0]["type"] == SUCCEEDED_CONDITION_TYPE
        )
        assert (
            result["status"]["status_conditions"][0]["status"] == CONDITION_STATUS_TRUE
        )

    def test_empty_conditions(self):
        job = _make_spark_job()
        result = _spark_job_to_dict(job)
        assert result["status"]["status_conditions"] == []


class TestCreateSparkJob:
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_creates_job_with_all_fields(self, mock_client):
        returned_job = _make_spark_job()
        mock_client.SparkJobService.create_spark_job.return_value = returned_job

        result = create_spark_job(
            namespace="my-ns",
            main_application_file="s3://bucket/app.jar",
            main_class="com.example.Main",
            args=["--date", "2026-01-01"],
            driver_cpu=2,
            driver_memory="4G",
            executor_cpu=4,
            executor_memory="8G",
            executor_instances=10,
            spark_conf={"spark.sql.shuffle.partitions": "200"},
            deps_jars=["s3://bucket/dep.jar"],
            deps_py_files=["s3://bucket/utils.py"],
            spark_version="3.5.5",
        )

        assert result == returned_job
        mock_client.SparkJobService.create_spark_job.assert_called_once()
        call_kwargs = mock_client.SparkJobService.create_spark_job.call_args
        spark_job = call_kwargs.kwargs["spark_job"]

        assert spark_job.metadata.namespace == "my-ns"
        assert spark_job.metadata.generateName == "uniflow-splg-"
        assert spark_job.spec.main_application_file == "s3://bucket/app.jar"
        assert spark_job.spec.main_class == "com.example.Main"
        assert list(spark_job.spec.main_args) == ["--date", "2026-01-01"]
        assert spark_job.spec.driver.pod.resource.cpu == 2
        assert spark_job.spec.driver.pod.resource.memory == "4G"
        assert spark_job.spec.executor.pod.resource.cpu == 4
        assert spark_job.spec.executor.pod.resource.memory == "8G"
        assert spark_job.spec.executor.instances == 10
        assert spark_job.spec.spark_conf["spark.sql.shuffle.partitions"] == "200"
        assert list(spark_job.spec.deps.jars) == ["s3://bucket/dep.jar"]
        assert list(spark_job.spec.deps.py_files) == ["s3://bucket/utils.py"]
        assert spark_job.spec.spark_version == "3.5.5"

    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_creates_job_minimal_fields(self, mock_client):
        returned_job = _make_spark_job()
        mock_client.SparkJobService.create_spark_job.return_value = returned_job

        result = create_spark_job(
            namespace="my-ns",
            main_application_file="s3://bucket/script.py",
        )

        assert result == returned_job
        call_kwargs = mock_client.SparkJobService.create_spark_job.call_args
        spark_job = call_kwargs.kwargs["spark_job"]
        assert spark_job.spec.main_class == ""
        assert list(spark_job.spec.main_args) == []
        assert spark_job.spec.driver.pod.resource.cpu == 0
        assert spark_job.spec.driver.pod.resource.memory == ""


class TestPollSparkJob:
    @patch("michelangelo.uniflow.core.lib.spark.job.time")
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_succeeds_immediately(self, mock_client, mock_time):
        mock_time.time.side_effect = [0.0, 0.1]
        succeeded_job = _make_spark_job(conditions=[_succeeded_condition()])
        mock_client.SparkJobService.get_spark_job.return_value = succeeded_job

        result = poll_spark_job("test-ns", "test-job")

        assert result["metadata"]["name"] == "test-job"
        assert (
            result["status"]["status_conditions"][0]["type"] == SUCCEEDED_CONDITION_TYPE
        )

    @patch("michelangelo.uniflow.core.lib.spark.job.time")
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_polls_then_succeeds(self, mock_client, mock_time):
        mock_time.time.side_effect = [0.0, 1.0, 2.0]
        running_job = _make_spark_job()
        succeeded_job = _make_spark_job(conditions=[_succeeded_condition()])
        mock_client.SparkJobService.get_spark_job.side_effect = [
            running_job,
            succeeded_job,
        ]

        result = poll_spark_job("test-ns", "test-job", timeout_seconds=60)

        assert (
            result["status"]["status_conditions"][0]["type"] == SUCCEEDED_CONDITION_TYPE
        )
        assert mock_client.SparkJobService.get_spark_job.call_count == 2

    @patch("michelangelo.uniflow.core.lib.spark.job.time")
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_raises_on_killed(self, mock_client, mock_time):
        mock_time.time.side_effect = [0.0, 0.1]
        killed_job = _make_spark_job(conditions=[_killed_condition()])
        mock_client.SparkJobService.get_spark_job.return_value = killed_job

        with pytest.raises(RuntimeError, match="was killed"):
            poll_spark_job("test-ns", "test-job")

    @patch("michelangelo.uniflow.core.lib.spark.job.time")
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_raises_on_failed(self, mock_client, mock_time):
        mock_time.time.side_effect = [0.0, 0.1]
        failed_job = _make_spark_job(conditions=[_failed_condition()])
        mock_client.SparkJobService.get_spark_job.return_value = failed_job

        with pytest.raises(RuntimeError, match="failed"):
            poll_spark_job("test-ns", "test-job")

    @patch("michelangelo.uniflow.core.lib.spark.job.time")
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_raises_on_timeout(self, mock_client, mock_time):
        mock_time.time.side_effect = [0.0, 100.0]
        mock_client.SparkJobService.get_spark_job.return_value = _make_spark_job()

        with pytest.raises(TimeoutError, match="timed out"):
            poll_spark_job("test-ns", "test-job", timeout_seconds=10)

    @patch("michelangelo.uniflow.core.lib.spark.job.time")
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_raises_on_not_found(self, mock_client, mock_time):
        mock_time.time.side_effect = [0.0, 0.1]
        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.NOT_FOUND
        rpc_error.details = lambda: "not found"
        mock_client.SparkJobService.get_spark_job.side_effect = rpc_error

        with pytest.raises(RuntimeError, match="not found"):
            poll_spark_job("test-ns", "test-job")

    @patch("michelangelo.uniflow.core.lib.spark.job.time")
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_retries_on_transient_error(self, mock_client, mock_time):
        mock_time.time.side_effect = [0.0, 1.0, 2.0]
        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.UNAVAILABLE
        rpc_error.details = lambda: "unavailable"
        succeeded_job = _make_spark_job(conditions=[_succeeded_condition()])
        mock_client.SparkJobService.get_spark_job.side_effect = [
            rpc_error,
            succeeded_job,
        ]

        result = poll_spark_job("test-ns", "test-job", timeout_seconds=60)
        assert result["metadata"]["name"] == "test-job"

    @patch("michelangelo.uniflow.core.lib.spark.job.time")
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_raises_on_permission_denied(self, mock_client, mock_time):
        mock_time.time.side_effect = [0.0, 0.1]
        rpc_error = grpc.RpcError()
        rpc_error.code = lambda: grpc.StatusCode.PERMISSION_DENIED
        rpc_error.details = lambda: "permission denied"
        mock_client.SparkJobService.get_spark_job.side_effect = rpc_error

        with pytest.raises(RuntimeError, match="permission denied"):
            poll_spark_job("test-ns", "test-job")


class TestCreateJob:
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_returns_dict(self, mock_client):
        returned_job = _make_spark_job(conditions=[_succeeded_condition()])
        mock_client.SparkJobService.create_spark_job.return_value = returned_job

        result = create_job(
            namespace="my-ns",
            main_application_file="s3://bucket/app.jar",
            main_class="com.example.Main",
        )

        assert isinstance(result, dict)
        assert result["metadata"]["name"] == "test-job"
        assert result["metadata"]["namespace"] == "test-ns"


class TestSensorJob:
    @patch("michelangelo.uniflow.core.lib.spark.job.time")
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_default_timeout(self, mock_client, mock_time):
        mock_time.time.side_effect = [0.0, 0.1]
        succeeded_job = _make_spark_job(conditions=[_succeeded_condition()])
        mock_client.SparkJobService.get_spark_job.return_value = succeeded_job

        result = sensor_job(namespace="test-ns", name="test-job")

        assert result["metadata"]["name"] == "test-job"

    @patch("michelangelo.uniflow.core.lib.spark.job.time")
    @patch("michelangelo.uniflow.core.lib.spark.job.APIClient")
    def test_custom_timeout(self, mock_client, mock_time):
        mock_time.time.side_effect = [0.0, 0.1]
        succeeded_job = _make_spark_job(conditions=[_succeeded_condition()])
        mock_client.SparkJobService.get_spark_job.return_value = succeeded_job

        result = sensor_job(
            namespace="test-ns",
            name="test-job",
            timeout_seconds=3600,
            poll_seconds=5,
        )

        assert result["metadata"]["name"] == "test-job"
