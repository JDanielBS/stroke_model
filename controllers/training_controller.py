"""
TrainingController
==================
Orquesta el pipeline completo de entrenamiento y la persistencia del modelo:

    1. Invoca DataPreprocessor.fit_transform_full() sobre el CSV.
    2. Entrena los 5 clasificadores mediante StrokeClassifier.train_all().
    3. Guarda el pipeline (preprocesador + PCA) y el mejor modelo.
    4. Devuelve el TrainingResponse con las métricas comparativas.

También expone métodos para cargar modelos ya guardados y consultar
el estado e información del modelo en uso.
"""

from pathlib import Path

from stroke_api.config              import DATA_PATH, MODELS_DIR
from stroke_api.models.preprocessor import DataPreprocessor
from stroke_api.models.classifier   import StrokeClassifier
from stroke_api.models.schemas      import (
    TrainingResponse,
    ModelInfoResponse,
    MetricEntry,
)


class TrainingController:
    """
    Controlador de entrenamiento y gestión del ciclo de vida del modelo.

    Mantiene referencias compartidas al preprocesador y al clasificador
    para que PredictionController pueda usarlos después del entrenamiento
    o carga.

    Parameters
    ----------
    preprocessor : DataPreprocessor
        Instancia compartida con PredictionController.
    classifier : StrokeClassifier
        Instancia compartida con PredictionController.
    """

    def __init__(
        self,
        preprocessor: DataPreprocessor,
        classifier:   StrokeClassifier,
    ) -> None:
        self._preprocessor = preprocessor
        self._classifier   = classifier

    # ─── Métodos públicos ──────────────────────────────────────────────────────

    def train(self, data_path: Path = DATA_PATH) -> TrainingResponse:
        """
        Ejecuta el pipeline completo de entrenamiento desde el CSV.

        Pasos internos
        --------------
        1. fit_transform_full  → limpieza, encoding, split, scaling, PCA, SMOTE
        2. train_all           → entrena y evalúa los 5 clasificadores
        3. save pipeline + model

        Parameters
        ----------
        data_path : Path, opcional
            Ruta al archivo CSV. Por defecto usa DATA_PATH de config.

        Returns
        -------
        TrainingResponse con métricas y rutas guardadas.
        """
        # ── Preprocesamiento ──────────────────────────────────────────────────
        X_train_bal, X_test_pca, y_train_bal, y_test = (
            self._preprocessor.fit_transform_full(data_path)
        )

        # ── Entrenamiento ─────────────────────────────────────────────────────
        all_metrics = self._classifier.train_all(
            X_train_bal, y_train_bal,
            X_test_pca,  y_test,
        )

        # ── Guardar artefactos ────────────────────────────────────────────────
        pipeline_saved = False
        model_saved    = False

        try:
            self._preprocessor.save(MODELS_DIR)
            pipeline_saved = True
        except Exception as exc:
            print(f"[TrainingController] No se pudo guardar el pipeline: {exc}")

        try:
            self._classifier.save(MODELS_DIR)
            model_saved = True
        except Exception as exc:
            print(f"[TrainingController] No se pudo guardar el modelo: {exc}")

        # ── Construir respuesta ───────────────────────────────────────────────
        best_name = self._classifier.best_model_name
        summary   = self._classifier.get_metrics_summary()

        return TrainingResponse(
            status          = "ok",
            best_model_name = best_name,
            best_auc_roc    = summary[best_name]["auc_roc"],
            best_recall     = summary[best_name]["recall"],
            metrics         = {
                name: MetricEntry(**vals) for name, vals in summary.items()
            },
            pipeline_saved  = pipeline_saved,
            model_saved     = model_saved,
        )

    def load_saved_model(self) -> ModelInfoResponse:
        """
        Carga desde disco el pipeline y el modelo previamente entrenados.
        Actualiza las instancias compartidas en memoria.

        Returns
        -------
        ModelInfoResponse con el estado y las métricas del modelo cargado.
        """
        loaded_preprocessor = DataPreprocessor.load(MODELS_DIR)
        loaded_classifier   = StrokeClassifier.load(MODELS_DIR)

        # Actualizar atributos en las instancias compartidas
        self._preprocessor.__dict__.update(loaded_preprocessor.__dict__)
        self._classifier.__dict__.update(loaded_classifier.__dict__)

        summary = self._classifier.get_metrics_summary()

        return ModelInfoResponse(
            is_loaded       = True,
            best_model_name = self._classifier.best_model_name,
            metrics_summary = {
                name: MetricEntry(**vals) for name, vals in summary.items()
            },
        )

    def get_model_info(self) -> ModelInfoResponse:
        """
        Devuelve información del modelo actualmente cargado en memoria.
        No realiza ninguna operación de I/O.
        """
        if not self._classifier.is_trained:
            return ModelInfoResponse(is_loaded=False)

        summary = self._classifier.get_metrics_summary()
        return ModelInfoResponse(
            is_loaded       = True,
            best_model_name = self._classifier.best_model_name,
            metrics_summary = {
                name: MetricEntry(**vals) for name, vals in summary.items()
            },
        )
