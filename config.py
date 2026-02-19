"""
Configuración global del proyecto Stroke Prediction API.
"""
from pathlib import Path

# ─── Rutas base ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent          # carpeta raíz del workspace
MODELS_DIR = BASE_DIR / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Dataset ───────────────────────────────────────────────────────────────────
DATA_PATH = DATA_DIR / "healthcare-dataset-stroke-data.csv"

# ─── Preprocesamiento ──────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
TARGET_COL  = "stroke"
DROP_COLS   = ["id"]
DROP_GENDER = ["Other"]          # filtramos la categoría rara

# ─── PCA ───────────────────────────────────────────────────────────────────────
PCA_VARIANCE_THRESHOLD = 0.90    # conservar el 90 % de la varianza

# ─── Entrenamiento ─────────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# Hiperparámetros por defecto para cada modelo
MODEL_PARAMS = {
    "knn": {
        "k_range": range(1, 30, 2),   # valores impares para evitar empates
        "metric": "euclidean",
    },
    "decision_tree": {
        "depth_range": range(2, 20),
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "adaboost": {
        "n_estimators": 200,
        "learning_rate": 0.5,
        "random_state": RANDOM_STATE,
        "base_max_depth": 2,
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "probability": True,
        "random_state": RANDOM_STATE,
    },
}

# ─── Serialización ─────────────────────────────────────────────────────────────
PIPELINE_FILENAME = "stroke_pipeline.joblib"   # preprocesador + PCA
MODEL_FILENAME     = "stroke_model.joblib"      # clasificador entrenado
METADATA_FILENAME  = "stroke_metadata.json"    # métricas y nombre del modelo

# ─── API ───────────────────────────────────────────────────────────────────────
API_TITLE       = "Stroke Prediction API"
API_DESCRIPTION = (
    "API para predecir el riesgo de stroke a partir de "
    "variables clínicas y demográficas del paciente."
)
API_VERSION = "1.0.0"
API_PREFIX  = "/api/v1"
