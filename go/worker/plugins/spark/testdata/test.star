load("@plugin", "spark")

def test_create_job():
    spec = {
        "kind": "SparkJob",
        "metadata": {
            "namespace": "test-namespace",
            "name": "test-name",
        },
        "spec": {
            "user": {
                "name": "michelangelo",
                "proxyUser": "michelangelo",
            },
            "affinity": {
                "resourceAffinity": {
                    "selector": {
                        "matchLabels": {
                            "resourcepool.michelangelo/zone": "dca60",
                            "resourcepool.michelangelo/cluster": "dca11-batch01",
                            "resourcepool.michelangelo/cluster_type": "peloton",
                            "resourcepool.michelangelo/name": "uberai-default-dca11-batch01",
                            "resourcepool.michelangelo/path": "/UberAI/Michelangelo/IntegrationTests",
                        },
                    },
                },
            },
            "driver": {
                "pod": {
                    "resource": {
                        "cpu": 2,
                        "memory": "8G",
                    },
                    "image": "127.0.0.1:5055/uber-system/sparkdocker:bkt1-produ-1698057657-38ca7",
                },
            },
            "executor": {
                "pod": {
                    "resource": {
                        "cpu": 2,
                        "memory": "8G",
                    },
                    "image": "127.0.0.1:5055/uber-system/sparkdocker:bkt1-produ-1698057657-38ca7",
                },
                "instances": 2,
            },
            "sparkConf": {
                "spark.peloton.run-as-user": "true",
                "spark.peloton.driver.docker.image": "127.0.0.1:5055/uber-system/sparkdocker:bkt1-produ-1698057657-38ca7",
                "spark.peloton.executor.docker.image": "127.0.0.1:5055/uber-system/sparkdocker:bkt1-produ-1698057657-38ca7",
            },
            "mainApplicationFile": "http://localhost:18839/prod/personal/andrii/pyspark_test.py",
            "mainArgs": ["--execution_run_id=test"],
            "deps": {},
            "scheduling": {},
            "sparkVersion": "SPARK_3",
        },
    }
    spark.create_job(job = spec)

def test_sensor_job():
    spec = {
        "kind": "SparkJob",
        "metadata": {
            "namespace": "test-namespace",
            "name": "test-name",
        },
    }

    spark.sensor_job(spec, assert_condition_type = "Succeeded")

def test_run_job():
    spark.run_job(
        namespace = "test-namespace",
        main_application_file = "local:///opt/spark/examples/jars/spark-examples.jar",
        main_class = "org.apache.spark.examples.SparkPi",
        args = ["2"],
        image = "apache/spark:3.5.5",
        driver_cpu = 1,
        driver_memory = "1G",
        executor_cpu = 1,
        executor_memory = "1G",
        executor_instances = 1,
        spark_version = "3.5.5",
    )
