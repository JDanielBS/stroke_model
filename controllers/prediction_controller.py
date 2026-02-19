"""
PredictionController
====================
Orquesta la lógica de predicción:

    1. Recibe el PatientRequest validado por Pydantic.
    2. Usa el DataPreprocessor cargado en memoria para transformar
       las variables del paciente al espacio PCA.
    3. Delega la predicción al StrokeClassifier.
    4. Construye y devuelve el PredictionResponse.

La instancia del controlador se comparte como singleton a través de
la inyección de dependencias de FastAPI (ver app.py).
"""

from stroke_api.models.preprocessor import DataPreprocessor
from stroke_api.models.classifier   import StrokeClassifier
from stroke_api.models.schemas      import PatientRequest, PredictionResponse


class PredictionController:
    """
    Controlador de predicciones de stroke.

    Parameters
    ----------
    preprocessor : DataPreprocessor
        Instancia ya cargada (ajustada) con los artefactos del pipeline.
    classifier : StrokeClassifier
        Instancia ya cargada con el mejor modelo entrenado.
    """

    def __init__(
        self,
        preprocessor: DataPreprocessor,
        classifier:   StrokeClassifier,
    ) -> None:
        self._preprocessor = preprocessor
        self._classifier   = classifier

    # ─── Métodos públicos ──────────────────────────────────────────────────────

    def predict(self, patient: PatientRequest) -> PredictionResponse:
        """
        Realiza la predicción de stroke para un único paciente.

        Parameters
        ----------
        patient : PatientRequest
            Datos del paciente validados por Pydantic.

        Returns
        -------
        PredictionResponse con el resultado de la predicción.

        Raises
        ------
        RuntimeError  si el preprocesador o el clasificador no están listos.
        ValueError    si algún valor de entrada es inválido para los encoders.
        """
        self._validate_ready()

        # Convertir el schema Pydantic a dict plano con los valores originales
        patient_dict = self._schema_to_dict(patient)

        # Transformar: encoding → scaling → PCA
        X_pca = self._preprocessor.transform(patient_dict)

        # Predicción
        result = self._classifier.predict(X_pca)

        return PredictionResponse(
            prediction       = result["prediction"],
            prediction_label = "Con Stroke" if result["prediction"] == 1 else "Sin Stroke",
            probability      = result["probability"],
            risk_level       = result["risk_level"],
            model_used       = result["model_used"],
        )

    def is_ready(self) -> bool:
        """Devuelve True si el preprocesador y el clasificador están listos."""
        return (
            self._preprocessor.is_fitted
            and self._classifier.is_trained
        )

    # ─── Métodos privados ──────────────────────────────────────────────────────

    def _validate_ready(self) -> None:
        if not self.is_ready():
            raise RuntimeError(
                "El modelo no está listo. "
                "Ejecuta el endpoint /train o carga un modelo guardado."
            )

    @staticmethod
    def _schema_to_dict(patient: PatientRequest) -> dict:
        """
        Convierte el schema Pydantic a un diccionario con los valores
        en el formato esperado por los LabelEncoders (strings originales).
        """
        return {
            "gender":            patient.gender.value,
            "age":               patient.age,
            "hypertension":      patient.hypertension,
            "heart_disease":     patient.heart_disease,
            "ever_married":      patient.ever_married.value,
            "work_type":         patient.work_type.value,
            "Residence_type":    patient.Residence_type.value,
            "avg_glucose_level": patient.avg_glucose_level,
            "bmi":               patient.bmi,
            "smoking_status":    patient.smoking_status.value,
        }
