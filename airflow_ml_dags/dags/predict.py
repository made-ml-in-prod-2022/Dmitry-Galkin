from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "d.galkin",
    "email": ["d.galkin.89@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def get_model_id() -> str:
    return Variable.get("model_id")


with DAG(
        dag_id="prediction",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(21),
) as dag:

    # id (дата) используемой модели
    model_id = get_model_id()

    start = DummyOperator(task_id="start")

    wait_for_data = FileSensor(
        task_id="wait-for-data",
        poke_interval=10,
        retries=5,
        filepath="data/raw/{{ ds }}/data.csv",
        email_on_failure=True,
    )

    wait_for_model = FileSensor(
        task_id="wait-for-model",
        poke_interval=10,
        retries=5,
        filepath=f"data/models/{model_id}/model.pkl",
        email_on_failure=True,
    )

    command = "--input-dir /data/raw/{{ ds }} --output-dir /data/predictions/{{ ds }} "
    command += f"--model-dir /data/models/{model_id}"
    predict = DockerOperator(
        image="airflow-predict",
        command=command,
        task_id="docker-airflow-predict",
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

    start >> [wait_for_data, wait_for_model] >> predict >> finish
