"""Tests that run_spark_job transpiles correctly inside a @workflow body."""

import tests.uniflow.core.test_spark_transpile_workflow as wf_module
from michelangelo.uniflow.core.build import build


class TestRunSparkJobTranspilation:
    """Verify @star_plugin('spark.run_job') rewrites to __spark__.run_job."""

    def test_transpiles_to_spark_run_job(self):
        """Verify run_spark_job is rewritten to __spark__.run_job in Starlark output."""
        package = build(wf_module.spark_pi_workflow)

        main_content = package.files[package.main_file]
        decoded = main_content.decode("utf-8")

        assert "__spark__.run_job(" in decoded
        assert "load('@plugin', __spark__='spark')" in decoded
        assert "run_spark_job" not in decoded

    def test_preserves_keyword_arguments(self):
        """Verify all keyword arguments are preserved in the transpiled output."""
        package = build(wf_module.spark_pi_workflow)

        main_content = package.files[package.main_file]
        decoded = main_content.decode("utf-8")

        assert "namespace='default'" in decoded
        assert "main_class='org.apache.spark.examples.SparkPi'" in decoded
        assert "main_application_file=" in decoded
        assert "image='apache/spark:3.5.5'" in decoded
        assert "driver_cpu=1" in decoded
