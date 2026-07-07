"""Spark job builtins for Uniflow workflows.

Provides pure-Python equivalents of the Go/Starlark spark.create_job and
spark.sensor_job builtins, using APIClient.SparkJobService directly.
"""

from michelangelo.uniflow.core.lib.spark.job import (
    KILLED_CONDITION_TYPE,
    SUCCEEDED_CONDITION_TYPE,
    create_job,
    create_spark_job,
    poll_spark_job,
    sensor_job,
)

__all__ = [
    "KILLED_CONDITION_TYPE",
    "SUCCEEDED_CONDITION_TYPE",
    "create_job",
    "create_spark_job",
    "poll_spark_job",
    "sensor_job",
]
