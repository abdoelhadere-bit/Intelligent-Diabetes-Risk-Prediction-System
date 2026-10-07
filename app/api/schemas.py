from pydantic import BaseModel, ConfigDict, Field


class PatientData(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"Glucose": 148, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627,
                        "Age": 50, "Insulin": 160}
        }
    )

    Glucose: float = Field(..., ge=40, le=300, description="Glycémie à 2 h (mg/dL)")
    BMI: float = Field(..., ge=10, le=80, description="Indice de masse corporelle (kg/m²)")
    DiabetesPedigreeFunction: float = Field(..., gt=0, le=3, description="Score d'antécédents familiaux")
    Age: int = Field(..., ge=21, le=100, description="Âge (ans) — modèle entraîné sur des adultes de 21 ans et plus")
    Insulin: float = Field(..., gt=0, le=1000, description="Insuline sérique à 2 h (µU/mL)")


class PredictionResponse(BaseModel):

    is_high_risk: bool
    risk_level: str
    probability_high_risk: float
    model_version: str