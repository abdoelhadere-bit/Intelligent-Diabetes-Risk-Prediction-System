from __future__ import annotations

import logging

import joblib
import matplotlib

matplotlib.use("Agg")  # pas d'écran dans un conteneur
import matplotlib.pyplot as plt  
import mlflow  
import numpy as np  
import pandas as pd  
import skops.io as sio  
from imblearn.pipeline import Pipeline as ImbPipeline  
from mlflow import MlflowClient  
from mlflow.models import infer_signature  
from sklearn.compose import ColumnTransformer  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,  
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import FunctionTransformer, StandardScaler  

from src.common import (CLASSIF_RESULTS, CLUSTER_STATE, CLUSTERED, HIGH_RISK_LABEL, LOW_RISK_LABEL,
                        MODELS_DIR, RANDOM_STATE, SNAPSHOT, TEST_SET, load_json,
                        log_reproducibility_info, read_csv, save_json, setup_mlflow)
from src.config import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, SELECTED_FEATURES
from src.data_preprocessing import LOG_COLUMNS

logger = logging.getLogger("classification")

TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = ["accuracy", "precision", "recall", "f1"]

PARAM_GRIDS = {
    "Logistic": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "KNN_classifier": {"model__n_neighbors": [3, 5, 7, 9, 11, 15],
                       "model__weights": ["uniform", "distance"]},
    "Random Forest": {"model__n_estimators": [100, 200],
                      "model__max_depth": [None, 5, 10],
                      "model__min_samples_leaf": [1, 3, 5]},
}


def make_models() -> dict:
    return {
        "Logistic": LogisticRegression(random_state=RANDOM_STATE),
        "KNN_classifier": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    }


def build_pipeline(model) -> ImbPipeline:
    # log -> standardisation -> modèle
    log_step = ColumnTransformer(
        [("log", FunctionTransformer(np.log), LOG_COLUMNS)], remainder="passthrough"
    )
    return ImbPipeline([("log", log_step), ("scaler", StandardScaler()), ("model", model)])


def train_classifiers() -> None:
    df = read_csv(CLUSTERED)
    raw_df = read_csv(SNAPSHOT)
    high_risk_cluster = load_json(CLUSTER_STATE)["high_risk_cluster"]

    X = df[SELECTED_FEATURES]
    y = (df["Cluster"] == high_risk_cluster).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_test.assign(target=y_test).to_csv(TEST_SET)  # réutilisé pour comparer à la prod

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    signature = infer_signature(X_train, y_train)
    results = {}

    setup_mlflow()
    for name, model in make_models().items():
        with mlflow.start_run(run_name=f"classification_{name}") as run:
            log_reproducibility_info(raw_df)
            grid = GridSearchCV(build_pipeline(model), param_grid=PARAM_GRIDS[name],
                                scoring=SCORING, refit="f1", cv=cv, n_jobs=-1).fit(X_train, y_train)

            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_dict({k: [str(v) for v in vals] for k, vals in PARAM_GRIDS[name].items()},
                            "param_grid.json")
            mlflow.log_params({f"best_{k.replace('model__', '')}": v
                               for k, v in grid.best_params_.items()})

            for i, params in enumerate(grid.cv_results_["params"]):
                with mlflow.start_run(run_name=f"{name}_combo_{i}", nested=True):
                    mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
                    mlflow.log_metrics({f"cv_{m}": grid.cv_results_[f"mean_test_{m}"][i]
                                        for m in SCORING})

            y_pred = grid.predict(X_test)
            metrics = {
                "cv_f1_mean": float(grid.best_score_),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_pred, zero_division=0),
                "test_f1": f1_score(y_test, y_pred, zero_division=0),
            }
            mlflow.log_metrics(metrics)

            display = ConfusionMatrixDisplay.from_estimator(
                grid.best_estimator_, X_test, y_test,
                display_labels=[LOW_RISK_LABEL, HIGH_RISK_LABEL], cmap="Blues", colorbar=False)
            display.ax_.set_title(f"Matrice de confusion — {name}")
            mlflow.log_figure(display.figure_, "confusion_matrix.png")
            plt.close(display.figure_)

            mlflow.sklearn.log_model(
                grid.best_estimator_, name="pipeline", signature=signature,
                input_example=X_train.head(3),
                skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(grid.best_estimator_)),
            )
            results[name] = {"run_id": run.info.run_id, **{k: float(v) for k, v in metrics.items()}}
            logger.info("%s : %s", name, {k: round(v, 3) for k, v in metrics.items()})

    save_json(CLASSIF_RESULTS, results)


# ----------------------------------------------------------------------------- Model Registry
def promote_to_production(client: MlflowClient, model_name: str, version: str) -> None:
    """Archive l'ancienne version Production, puis promeut la nouvelle (stage + alias)."""
    for existing in client.search_model_versions(f"name='{model_name}'"):
        if existing.current_stage == "Production" and existing.version != str(version):
            client.transition_model_version_stage(model_name, existing.version, stage="Archived")
    client.transition_model_version_stage(model_name, version, stage="Production")
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version)


def _production_f1(X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    """F1 du modèle actuellement en production, sur le NOUVEAU jeu de test (comparaison équitable)."""
    try:
        current = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}")
    except Exception:
        logger.info("Aucun modèle en production pour l'instant.")
        return None
    return float(f1_score(y_test, current.predict(X_test), zero_division=0))


def evaluate_and_register() -> None:
    results = load_json(CLASSIF_RESULTS)
    test = read_csv(TEST_SET)
    X_test, y_test = test[SELECTED_FEATURES], test["target"]

    # Meilleur modèle : F1 en validation croisée (plus fiable que le test), puis F1 test
    best_name = max(results, key=lambda n: (results[n]["cv_f1_mean"], results[n]["test_f1"]))
    best = results[best_name]
    logger.info("Meilleur modèle : %s (cv_f1=%.4f, test_f1=%.4f)",
                best_name, best["cv_f1_mean"], best["test_f1"])

    setup_mlflow()
    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{best['run_id']}/pipeline", REGISTERED_MODEL_NAME)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "model_type", best_name)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "test_f1", f"{best['test_f1']:.4f}")

    prod_f1 = _production_f1(X_test, y_test)
    if prod_f1 is None or best["test_f1"] >= prod_f1:
        promote_to_production(client, REGISTERED_MODEL_NAME, version.version)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"),
                    MODELS_DIR / "pipeline.joblib")
        logger.info("Version %s promue en Production (F1 %.4f vs prod %s)",
                    version.version, best["test_f1"], prod_f1)
    else:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", version.version)
        logger.warning("Version %s NON promue : F1 %.4f < prod %.4f (alias @challenger posé)",
                       version.version, best["test_f1"], prod_f1)