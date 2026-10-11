from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/project"
PYTHON = "/home/airflow/ml-venv/bin/python"

default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def step(task_id: str, cli_step: str, **kwargs) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} -m src.training_pipeline {cli_step}",
        env={"PYTHONPATH": PROJECT_DIR, "PYTHONUNBUFFERED": "1"},
        append_env=True,  # garde MLFLOW_TRACKING_URI, API_URL, PROJECT_ROOT du conteneur
        **kwargs,
    )


with DAG(
    dag_id="diabetes_risk_training",
    description="Extraction, prétraitement, clustering, classification, mise à jour du Model Registry",
    start_date=datetime(2026, 10, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["diabetes", "mlflow", "retraining"],
) as dag:
    extract = step("extract_data", "extract")
    preprocess = step("preprocess", "preprocess")
    clustering = step("train_clustering", "clustering")
    classifiers = step("train_classifiers", "classifiers", execution_timeout=timedelta(hours=1))
    register = step("evaluate_and_register", "register")
    reload_api = step("reload_api", "reload_api")

    extract >> preprocess >> clustering >> classifiers >> register >> reload_api