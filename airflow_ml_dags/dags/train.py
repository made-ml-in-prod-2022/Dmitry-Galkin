from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "d.galkin",
    "email": ["d.galkin.89@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


with DAG(
        dag_id="train",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(21),
) as dag:

    start = DummyOperator(task_id="start")

    wait_for_features = FileSensor(
        task_id="wait-for-features",
        poke_interval=10,
        retries=5,
        filepath="data/raw/{{ ds }}/data.csv",
        email_on_failure=True,
    )

    wait_for_target = FileSensor(
        task_id="wait-for-target",
        poke_interval=10,
        retries=5,
        filepath="data/raw/{{ ds }}/target.csv",
        email_on_failure=True,
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} "
                "--output-dir /data/processed/{{ ds }}",
        task_id="docker-airflow-preprocess",
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

    split = DockerOperator(
        image="airflow-split",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/split/{{ ds }}",
        task_id="docker-airflow-split",
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

    train = DockerOperator(
        image="airflow-train",
        command="--input-dir /data/split/{{ ds }} --output-dir /data/models/{{ ds }}",
        task_id="docker-airflow-train",
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

    validate = DockerOperator(
        image="airflow-validate",
        command="--input-dir /data/split/{{ ds }} --model-dir /data/models/{{ ds }}",
        task_id="docker-airflow-validate",
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

    finish = DummyOperator(task_id="finish")

    start >> [wait_for_features, wait_for_target] >> preprocess
    preprocess >> split >> train >> validate >> finish
