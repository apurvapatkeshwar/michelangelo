"""Minimal test workflow that calls run_spark_job directly in a @workflow body.

Used to verify the transpiler rewrites run_spark_job to __spark__.run_job.
"""

from michelangelo.uniflow.core import workflow
from michelangelo.uniflow.core.lib.spark.job import run_spark_job


@workflow()
def spark_pi_workflow():
    """Submit SparkPi and wait for completion."""
    return run_spark_job(
        namespace="default",
        main_application_file="local:///opt/spark/examples/jars/spark-examples.jar",
        main_class="org.apache.spark.examples.SparkPi",
        args=["2"],
        image="apache/spark:3.5.5",
        driver_cpu=1,
        driver_memory="1g",
        executor_cpu=1,
        executor_memory="1g",
        executor_instances=1,
    )
