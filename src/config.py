import os

# Variables d'entrée du modèle
SELECTED_FEATURES = ["Glucose", "BMI", "DiabetesPedigreeFunction", "Age", "Insulin"]

# MLflow 
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = "diabetes-risk-pipeline"
PRODUCTION_ALIAS = "production"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"