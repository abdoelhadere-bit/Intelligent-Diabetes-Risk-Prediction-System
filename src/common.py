from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from pathlib import Path

import imblearn
import mlflow
import numpy as np
import pandas as pd
import sklearn

from src.config import MLFLOW_TRACKING_URI as DEFAULT_TRACKING_URI
from src.config import SELECTED_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# --- Chemins & environnement ---
ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RAW_FILE = Path(os.getenv("DATA_FILE", ROOT / "data" / "raw" / "dataset-diabete.csv"))
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "diabetes-risk-prediction")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

RANDOM_STATE = 42
HIGH_RISK_LABEL, LOW_RISK_LABEL = "Risque élevé", "Risque faible"

# --- Fichiers intermédiaires échangés entre les tâches Airflow ---
SNAPSHOT = PROCESSED_DIR / "raw_snapshot.csv"
IMPUTED = PROCESSED_DIR / "df_impute.csv"
TRANSFORMED = PROCESSED_DIR / "df_transformed.csv"
CLUSTERED = PROCESSED_DIR / "df_clustered.csv"
CLUSTER_STATE = PROCESSED_DIR / "clustering_state.json"
TEST_SET = PROCESSED_DIR / "test_set.csv"
CLASSIF_RESULTS = PROCESSED_DIR / "classification_results.json"


def to_dataframe(array: np.ndarray, reference: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(array, columns=reference.columns, index=reference.index)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : l'étape précédente a-t-elle été exécutée ?")
    return json.loads(path.read_text())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : l'étape précédente a-t-elle été exécutée ?")
    return pd.read_csv(path, index_col=0)


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_reproducibility_info(raw_df: pd.DataFrame) -> None:
    mlflow.log_dict({"selected_features": SELECTED_FEATURES}, "features.json")
    mlflow.log_params({
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "imblearn_version": imblearn.__version__,
    })
    mlflow.log_param("dataset_sha256", hashlib.sha256(RAW_FILE.read_bytes()).hexdigest())
    mlflow.log_input(mlflow.data.from_pandas(raw_df, source=str(RAW_FILE), name="diabetes_raw"),
                     context="training")