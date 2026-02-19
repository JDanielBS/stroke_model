"""
Schemas Pydantic
================
Define los modelos de datos para las peticiones y respuestas de la API.
Validan automáticamente entradas y generan la documentación OpenAPI.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─── Enumeraciones permitidas ──────────────────────────────────────────────────

class GenderEnum(str, Enum):
    male   = "Male"
    female = "Female"


class MarriedEnum(str, Enum):
    yes = "Yes"
    no  = "No"


class WorkTypeEnum(str, Enum):
    private       = "Private"
    self_employed = "Self-employed"
    govt_job      = "Govt_job"
    children      = "children"
    never_worked  = "Never_worked"


class ResidenceEnum(str, Enum):
    urban = "Urban"
    rural = "Rural"


class SmokingEnum(str, Enum):
    never_smoked    = "never smoked"
    formerly_smoked = "formerly smoked"
    smokes          = "smokes"
    unknown         = "Unknown"


class RiskLevelEnum(str, Enum):
    low      = "BAJO"
    moderate = "MODERADO"
    high     = "ALTO"


# ─── Petición de predicción ────────────────────────────────────────────────────

class PatientRequest(BaseModel):
    """
    Variables clínicas y demográficas de un paciente.
    Corresponden exactamente a las features del dataset de entrenamiento.
    """
    gender:            GenderEnum  = Field(..., description="Sexo biológico del paciente")
    age:               float       = Field(..., ge=0, le=120,  description="Edad en años")
    hypertension:      int         = Field(..., ge=0, le=1,    description="1 = tiene hipertensión")
    heart_disease:     int         = Field(..., ge=0, le=1,    description="1 = tiene enfermedad cardíaca")
    ever_married:      MarriedEnum = Field(..., description="¿Ha estado casado alguna vez?")
    work_type:         WorkTypeEnum = Field(..., description="Tipo de empleo")
    Residence_type:    ResidenceEnum = Field(..., description="Tipo de residencia")
    avg_glucose_level: float       = Field(..., ge=0,          description="Nivel promedio de glucosa en sangre")
    bmi:               float       = Field(..., ge=0, le=100,  description="Índice de masa corporal")
    smoking_status:    SmokingEnum = Field(..., description="Estado de tabaquismo")

    @field_validator("hypertension", "heart_disease")
    @classmethod
    def binary_flag(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("El valor debe ser 0 o 1.")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender":            "Male",
                    "age":               75,
                    "hypertension":      1,
                    "heart_disease":     1,
                    "ever_married":      "Yes",
                    "work_type":         "Self-employed",
                    "Residence_type":    "Urban",
                    "avg_glucose_level": 220.0,
                    "bmi":               32.0,
                    "smoking_status":    "formerly smoked",
                }
            ]
        }
    }


# ─── Respuestas ────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Resultado de la predicción para un paciente."""
    prediction:        int          = Field(..., description="0 = Sin Stroke, 1 = Con Stroke")
    prediction_label:  str          = Field(..., description="Etiqueta legible")
    probability:       float        = Field(..., description="Probabilidad de stroke (0–1)")
    risk_level:        RiskLevelEnum = Field(..., description="Nivel de riesgo calculado")
    model_used:        str          = Field(..., description="Nombre del modelo empleado")


class MetricEntry(BaseModel):
    """Métricas de un modelo individual."""
    accuracy:  float
    precision: float
    recall:    float
    f1_score:  float
    auc_roc:   float


class TrainingResponse(BaseModel):
    """Resultado del pipeline de entrenamiento completo."""
    status:          str
    best_model_name: str
    best_auc_roc:    float
    best_recall:     float
    metrics:         dict[str, MetricEntry]
    pipeline_saved:  bool
    model_saved:     bool


class ModelInfoResponse(BaseModel):
    """Información del modelo actualmente cargado en memoria."""
    is_loaded:       bool
    best_model_name: Optional[str] = None
    metrics_summary: Optional[dict[str, MetricEntry]] = None


class ErrorResponse(BaseModel):
    """Modelo estándar para respuestas de error."""
    detail: str
