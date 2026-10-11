from __future__ import annotations

import argparse
import logging

import requests

from src.classification import evaluate_and_register, train_classifiers
from src.clustering import train_clustering
from src.common import API_URL
from src.data_preprocessing import extract_data, preprocess

logger = logging.getLogger("training-pipeline")


def reload_api() -> None:
    try:
        resp = requests.post(f"{API_URL}/reload", timeout=30)
        resp.raise_for_status()
        logger.info("API rechargée : %s", resp.json())
    except requests.exceptions.RequestException as exc:
        logger.warning("Rechargement de l'API impossible (%s). Le modèle est quand même enregistré.", exc)


STEPS = {
    "extract": extract_data,
    "preprocess": preprocess,
    "clustering": train_clustering,
    "classifiers": train_classifiers,
    "register": evaluate_and_register,
    "reload_api": reload_api,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement diabète")
    parser.add_argument("step", choices=[*STEPS, "all"])
    args = parser.parse_args()
    for name in (STEPS if args.step == "all" else [args.step]):
        logger.info("=== Étape : %s ===", name)
        STEPS[name]()


if __name__ == "__main__":
    main()