import logging
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from app.api.schemas import PatientData, PredictionResponse
from src.config import (MLFLOW_TRACKING_URI, MODEL_URI, PRODUCTION_ALIAS,
                        REGISTERED_MODEL_NAME, SELECTED_FEATURES)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diabetes-api")

# État de l'application 
state = {"model": None, "version": None}


def load_production_model() -> None:
    """Charge le pipeline portant l'alias @production depuis le Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)
    state["version"] = MlflowClient().get_model_version_by_alias(
        REGISTERED_MODEL_NAME, PRODUCTION_ALIAS
    ).version
    logger.info("Modèle chargé : %s (version %s)", MODEL_URI, state["version"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Exécuté au démarrage : charge le modèle une seule fois."""
    try:
        load_production_model()
    except Exception:
        logger.exception("Impossible de charger le modèle au démarrage")
    yield


app = FastAPI(title="Diabetes Risk API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Indique si l'API est en ligne et si un modèle est chargé."""
    return {"status": "ok", "model_loaded": state["model"] is not None,
            "model_version": state["version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData) -> PredictionResponse:
    """Prédit le niveau de risque à partir des constantes cliniques."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible, réessayez plus tard.")

    features = pd.DataFrame([patient.model_dump()])[SELECTED_FEATURES]
    is_high_risk = bool(state["model"].predict(features)[0])
    probability = float(state["model"].predict_proba(features)[0][1])

    return PredictionResponse(
        is_high_risk=is_high_risk,
        risk_level="Risque élevé" if is_high_risk else "Risque faible",
        probability_high_risk=round(probability, 4),
        model_version=state["version"],
    )


@app.post("/reload")
def reload_model() -> dict:
    """Recharge le modèle @production (après un réentraînement par Airflow)."""
    try:
        load_production_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Rechargement impossible : {exc}") from exc
    return {"status": "reloaded", "model_version": state["version"]}