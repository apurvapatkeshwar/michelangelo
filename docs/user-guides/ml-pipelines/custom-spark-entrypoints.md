# Custom Spark Entrypoints with `run_spark_job()`

`run_spark_job()` submits a custom Spark entrypoint (a Scala/JVM jar or a standalone PySpark script) and waits for it to complete. It is callable directly inside a `@workflow()` body — no `@task` decorator or `TaskConfig` object needed.

## When to use `run_spark_job()` vs `SparkTask`

| | `SparkTask` | `run_spark_job()` |
|---|---|---|
| **Entrypoint** | Uniflow-wrapped Python function (`run_task.py`) | Your own jar or `.py` file |
| **Typed I/O** | Full — reads/writes through Uniflow's I/O registry | None — returns job status only |
| **Return value** | Marshaled Python return value from the task function | Terminal job status dict (metadata + status conditions) |
| **Use when** | Your task logic lives in Python and you want typed data passed between workflow steps | You have an existing Spark job (Scala/JVM or standalone PySpark) and don't need Uniflow I/O |

**Key trade-off:** `run_spark_job()` returns only the terminal job status — not a Python return value. If a downstream workflow step needs data the Spark job produced, that data must be written by the job itself to a known location (a table, an S3 path passed via `args` or `spark_conf`) and referenced explicitly in the next step.

## Usage

```python
import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.core.lib.spark import run_spark_job


@uniflow.workflow()
def my_pipeline(date: str):
    # Scala/JVM jar example
    result = run_spark_job(
        namespace="my-project",
        main_application_file="s3://my-bucket/my-app-1.0.jar",
        main_class="com.example.MySparkApp",
        args=["--date", date],
        image="apache/spark:3.5.5",
        executor_cpu=4,
        executor_memory="4G",
        executor_instances=10,
        retry_attempts=2,
    )
    return result
```

## Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `namespace` | `str` | Yes | Namespace where the SparkJob will be created |
| `main_application_file` | `str` | Yes | Path or URI to the entrypoint jar or `.py` file |
| `main_class` | `str` | No | Fully qualified main class (required for jar entrypoints, omit for `.py`) |
| `args` | `list[str]` | No | Arguments passed to the application |
| `image` | `str` | No | Container image for driver and executor pods |
| `driver_cpu` | `int` | No | CPU cores for the driver |
| `driver_memory` | `str` | No | Memory for the driver (e.g. `"4G"`) |
| `executor_cpu` | `int` | No | CPU cores per executor |
| `executor_memory` | `str` | No | Memory per executor (e.g. `"8G"`) |
| `executor_instances` | `int` | No | Number of executor instances |
| `spark_conf` | `dict[str, str]` | No | Spark configuration properties |
| `deps_jars` | `list[str]` | No | JAR dependencies |
| `deps_py_files` | `list[str]` | No | Python file dependencies |
| `spark_version` | `str` | No | Spark version (default: `"3.5.5"`) |
| `timeout_seconds` | `int` | No | Max wait time in seconds (default: 10 years) |
| `poll_seconds` | `int` | No | Poll interval in seconds (default: 10) |
| `retry_attempts` | `int` | No | Number of retries on failure (default: 0) |

## Return value

```python
{
    "metadata": {
        "name": "uniflow-splg-abc12",
        "namespace": "my-project"
    },
    "status": {
        "status_conditions": [
            {"type": "Succeeded", "status": 1, "reason": "...", "message": "..."}
        ],
        "job_url": "...",
        "application_id": "..."
    }
}
```

## Retry behavior

When `retry_attempts > 0`, `run_spark_job()` re-submits a fresh SparkJob on any non-succeeded terminal state, including both Failed and Killed conditions. Killed is retried because it is commonly caused by infrastructure instability (node preemption, resource pressure), not deliberate cancellation. Explicit run cancellation is handled separately by the workflow engine's own cancellation mechanism and is unaffected by `retry_attempts`.

## Accessing job outputs

Since `run_spark_job()` does not marshal a Python return value, downstream steps must access job outputs through external storage:

```python
@uniflow.workflow()
def my_pipeline():
    output_path = "s3://my-bucket/output/"
    run_spark_job(
        namespace="my-project",
        main_application_file="s3://my-bucket/etl.jar",
        main_class="com.example.ETL",
        args=["--output", output_path],
    )
    # Next step references the output location directly
    return process_results(output_path)
```
