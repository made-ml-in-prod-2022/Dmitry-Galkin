from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "d.galkin",
    "email": ["d.galkin.89@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
        dag_id="download_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(21)
) as dag:

    start_task = DummyOperator(task_id="start-download-data")

    download = DockerOperator(
        image="airflow-download",
        command="--output-dir /data/raw/{{ ds }} "
                "--data-url https://drive.google.com/uc?id=1H0IDV40zbiJlBh5O27yfTc3wZ-23eK3Z",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(
                source="C:/Users/Clover/dgalk/ml.ai/made/2-й семестр/ML в продакшене/"
                       "Dmitry-Galkin/airflow_ml_dags/data/",
                target="/data",
                type="bind"
            )
        ],
        email_on_failure=True,
    )

    end_task = DummyOperator(task_id="end-download-data")

    start_task >> download >> end_task
