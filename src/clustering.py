from __future__ import annotations

import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.common import (CLUSTER_STATE, CLUSTERED, HIGH_RISK_LABEL, IMPUTED, LOW_RISK_LABEL,
                        MODELS_DIR, RANDOM_STATE, SNAPSHOT, TRANSFORMED,
                        log_reproducibility_info, read_csv, save_json, setup_mlflow, to_dataframe)
from src.config import SELECTED_FEATURES
from src.data_preprocessing import LOG_COLUMNS

logger = logging.getLogger("clustering")

OPTIMAL_K = int(os.getenv("KMEANS_K", "2"))
K_RANGE = range(2, 11)
RISK_THRESHOLDS = {"Glucose": 126, "BMI": 30, "DiabetesPedigreeFunction": 0.5}


def train_clustering() -> None:
    df_impute, df_transformed = read_csv(IMPUTED), read_csv(TRANSFORMED)
    raw_df = read_csv(SNAPSHOT)

    X = df_transformed[SELECTED_FEATURES]
    cluster_scaler = StandardScaler()
    X_scaled = to_dataframe(cluster_scaler.fit_transform(X), X)

    # Balayage de k, archivé pour justifier le choix
    k_rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        k_rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X_scaled, labels)})
    k_scores = pd.DataFrame(k_rows).set_index("k")

    kmeans = KMeans(n_clusters=OPTIMAL_K, n_init=10, random_state=RANDOM_STATE)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, cluster_labels)


    df_clustered = df_impute.copy()
    df_clustered["Cluster"] = cluster_labels
    profile = df_clustered.groupby("Cluster")[SELECTED_FEATURES].mean()
    above = pd.DataFrame({f: profile[f] > t for f, t in RISK_THRESHOLDS.items()}).all(axis=1)
    high_risk_clusters = above.index[above].tolist()
    if len(high_risk_clusters) != 1:
        raise ValueError(f"Attendu 1 cluster à haut risque, trouvé {len(high_risk_clusters)} : "
                         f"{high_risk_clusters}\n{profile.round(2)}")
    high_risk_cluster = int(high_risk_clusters[0])
    df_clustered["risk_category"] = np.where(
        df_clustered["Cluster"] == high_risk_cluster, HIGH_RISK_LABEL, LOW_RISK_LABEL)
    n_high_risk = int((df_clustered["Cluster"] == high_risk_cluster).sum())
    logger.info("Cluster à haut risque : %d (%d patientes)", high_risk_cluster, n_high_risk)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cluster_scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(kmeans, MODELS_DIR / "kmeans.joblib")
    df_clustered.to_csv(CLUSTERED)
    save_json(CLUSTER_STATE, {"high_risk_cluster": high_risk_cluster})

    setup_mlflow()
    with mlflow.start_run(run_name="kmeans_clustering"):
        log_reproducibility_info(raw_df)
        mlflow.log_params({
            "k": OPTIMAL_K, "n_init": 10, "random_state": RANDOM_STATE,
            "scaler": "StandardScaler", "log_columns": str(LOG_COLUMNS),
            "high_risk_cluster": high_risk_cluster,
        })
        mlflow.log_params({f"scaler_mean_{f}": round(m, 4)
                           for f, m in zip(SELECTED_FEATURES, cluster_scaler.mean_)})
        mlflow.log_params({f"scaler_scale_{f}": round(s, 4)
                           for f, s in zip(SELECTED_FEATURES, cluster_scaler.scale_)})
        mlflow.log_metrics({"inertia": kmeans.inertia_, "silhouette": silhouette,
                            "High_risk": n_high_risk})
        mlflow.log_table(k_scores.reset_index(), "k_scores.json")
        mlflow.log_artifact(str(MODELS_DIR / "kmeans.joblib"), artifact_path="models")
        mlflow.log_artifact(str(MODELS_DIR / "scaler.joblib"), artifact_path="models")