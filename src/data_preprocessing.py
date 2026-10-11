from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from src.common import (IMPUTED, RAW_FILE, SNAPSHOT, TRANSFORMED, read_csv, to_dataframe)

logger = logging.getLogger("data-preprocessing")

MIN_ROWS = 100
EXPECTED_COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
# Colonnes pour lesquelles une valeur de 0 est physiologiquement impossible
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
MIN_BLOOD_PRESSURE = 40
MAX_SKIN_THICKNESS = 80
# Variables fortement asymétriques à droite -> transformation logarithmique
LOG_COLUMNS = ["Insulin", "DiabetesPedigreeFunction"]


def extract_data() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Jeu de données introuvable : {RAW_FILE}")
    df = pd.read_csv(RAW_FILE, index_col=0)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {sorted(missing)}")
    df = df[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="raise")

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Extraction : %d lignes lues, %d doublons supprimés", n_before, n_before - len(df))
    if len(df) < MIN_ROWS:
        raise ValueError(f"Trop peu de lignes ({len(df)} < {MIN_ROWS}) : ré-entraînement annulé.")

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SNAPSHOT)
    logger.info("Snapshot écrit : %s %s", SNAPSHOT, df.shape)


def preprocess() -> None:
    df = read_csv(SNAPSHOT)

    df_clean = df.copy()
    df_clean[ZERO_AS_MISSING] = df_clean[ZERO_AS_MISSING].replace(0, np.nan)
    df_clean.loc[df_clean["BloodPressure"] < MIN_BLOOD_PRESSURE, "BloodPressure"] = np.nan
    df_clean.loc[df_clean["SkinThickness"] > MAX_SKIN_THICKNESS, "SkinThickness"] = np.nan
    logger.info("Valeurs manquantes avant imputation :\n%s", df_clean.isna().sum().to_string())

    # KNN repose sur des distances 
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_clean)
    x_imputed = KNNImputer(n_neighbors=5).fit_transform(x_scaled)
    df_impute = to_dataframe(scaler.inverse_transform(x_imputed), df_clean)

    df_transformed = df_impute.copy()
    df_transformed[LOG_COLUMNS] = np.log(df_transformed[LOG_COLUMNS])

    if df_transformed.isna().any().any() or np.isinf(df_transformed.to_numpy()).any():
        raise ValueError("Valeurs NaN/inf après prétraitement.")

    df_impute.to_csv(IMPUTED)
    df_transformed.to_csv(TRANSFORMED)
    logger.info("Prétraitement terminé : %s", df_impute.shape)