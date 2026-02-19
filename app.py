"""
Stroke Prediction API — Punto de entrada FastAPI
=================================================
Crea la aplicación, inyecta los controladores como dependencias
y declara todos los endpoints disponibles.

Endpoints
---------
POST /api/v1/predict          →  Predecir riesgo de stroke para un paciente.
POST /api/v1/train            →  Entrenar (o re-entrenar) todos los modelos.
POST /api/v1/model/load       →  Cargar el modelo guardado desde disco.
GET  /api/v1/model/info       →  Consultar el modelo actualmente en memoria.
GET  /api/v1/health           →  Health-check de la API.

Ejecución
---------
    uvicorn stroke_api.app:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from stroke_api.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    API_PREFIX,
    MODELS_DIR,
    MODEL_FILENAME,
    PIPELINE_FILENAME,
)
from stroke_api.models.preprocessor          import DataPreprocessor
from stroke_api.models.classifier            import StrokeClassifier
from stroke_api.models.schemas               import (
    PatientRequest,
    PredictionResponse,
    TrainingResponse,
    ModelInfoResponse,
    ErrorResponse,
)
from stroke_api.controllers.prediction_controller import PredictionController
from stroke_api.controllers.training_controller   import TrainingController


# ─── Singleton de artefactos compartidos ──────────────────────────────────────
# Ambos controladores comparten las mismas instancias en memoria.
_preprocessor = DataPreprocessor()
_classifier   = StrokeClassifier()

prediction_ctrl = PredictionController(_preprocessor, _classifier)
training_ctrl   = TrainingController(_preprocessor, _classifier)


# ─── Lifespan: intenta cargar modelo guardado al arrancar ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Al iniciar la aplicación intenta cargar automáticamente el último
    pipeline y modelo guardados en disco (si existen).
    """
    pipeline_exists = (MODELS_DIR / PIPELINE_FILENAME).exists()
    model_exists    = (MODELS_DIR / MODEL_FILENAME).exists()

    if pipeline_exists and model_exists:
        try:
            training_ctrl.load_saved_model()
            print(
                f"[Startup] Modelo '{_classifier.best_model_name}' "
                "cargado automáticamente desde disco."
            )
        except Exception as exc:
            print(f"[Startup] No se pudo cargar el modelo guardado: {exc}")
    else:
        print(
            "[Startup] No se encontró modelo guardado. "
            "Usa POST /api/v1/train para entrenar."
        )

    yield  # La app queda disponible


# ─── Instancia FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(
    title       = API_TITLE,
    description = API_DESCRIPTION,
    version     = API_VERSION,
    lifespan    = lifespan,
)


# ─── Dependencias ─────────────────────────────────────────────────────────────
def get_prediction_controller() -> PredictionController:
    return prediction_ctrl


def get_training_controller() -> TrainingController:
    return training_ctrl


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get(
    f"{API_PREFIX}/health",
    summary="Health-check",
    tags=["Utilidades"],
)
def health_check():
    """Verifica que la API esté activa y si hay un modelo listo."""
    return {
        "status":      "ok",
        "model_ready": prediction_ctrl.is_ready(),
        "model_name":  _classifier.best_model_name or None,
    }


@app.post(
    f"{API_PREFIX}/predict",
    response_model   = PredictionResponse,
    responses        = {503: {"model": ErrorResponse}},
    summary          = "Predecir riesgo de stroke",
    tags             = ["Predicción"],
)
def predict(
    patient:  PatientRequest,
    ctrl:     PredictionController = Depends(get_prediction_controller),
):
    """
    Recibe los datos clínicos y demográficos de un paciente y devuelve
    la probabilidad estimada de sufrir un stroke junto con su nivel de riesgo.

    El modelo debe haber sido entrenado (POST /train) o cargado
    (POST /model/load) previamente.
    """
    try:
        return ctrl.predict(patient)
    except RuntimeError as exc:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = str(exc),
        )


@app.post(
    f"{API_PREFIX}/train",
    response_model   = TrainingResponse,
    responses        = {500: {"model": ErrorResponse}},
    summary          = "Entrenar / re-entrenar los modelos",
    tags             = ["Entrenamiento"],
)
def train(
    ctrl: TrainingController = Depends(get_training_controller),
):
    """
    Ejecuta el pipeline de entrenamiento completo:

    1. Carga el CSV desde la ruta configurada en `config.DATA_PATH`.
    2. Aplica limpieza, encoding, escala, PCA y SMOTE.
    3. Entrena KNN, Árbol de Decisión, Random Forest, AdaBoost y SVM.
    4. Selecciona el mejor modelo por AUC-ROC.
    5. Guarda pipeline y modelo en `saved_models/`.

    ⚠️ Este proceso puede tardar varios minutos dependiendo del hardware.
    """
    try:
        return ctrl.train()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"Archivo de datos no encontrado: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Error durante el entrenamiento: {exc}",
        )


@app.post(
    f"{API_PREFIX}/model/load",
    response_model   = ModelInfoResponse,
    responses        = {404: {"model": ErrorResponse}},
    summary          = "Cargar el modelo guardado desde disco",
    tags             = ["Modelo"],
)
def load_model(
    ctrl: TrainingController = Depends(get_training_controller),
):
    """
    Carga en memoria el pipeline y el modelo previamente entrenados y
    guardados en `saved_models/`. Útil para restaurar el modelo tras un
    reinicio de la API sin necesidad de re-entrenar.
    """
    try:
        return ctrl.load_saved_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = str(exc),
        )


@app.get(
    f"{API_PREFIX}/model/info",
    response_model   = ModelInfoResponse,
    summary          = "Información del modelo en memoria",
    tags             = ["Modelo"],
)
def model_info(
    ctrl: TrainingController = Depends(get_training_controller),
):
    """
    Devuelve el nombre y las métricas comparativas (accuracy, precision,
    recall, F1, AUC-ROC) de todos los modelos entrenados en la sesión actual.
    """
    return ctrl.get_model_info()
