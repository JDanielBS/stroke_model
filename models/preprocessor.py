"""
DataPreprocessor
================
Responsable de todo el pipeline de preprocesamiento que precede al
entrenamiento del modelo:

    Carga de CSV
        → Limpieza inicial (drop 'id', filtrar género 'Other')
        → Imputación BMI con mediana
        → LabelEncoding de variables categóricas
        → Train/Test split (estratificado)
        → StandardScaler
        → PCA (umbral de varianza configurable)
        → SMOTE sobre el conjunto de entrenamiento

Los artefactos ajustados (encoders, scaler, pca) se guardan internamente y
pueden ser serializados/cargados con joblib para inferencia en producción.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from stroke_api.config import (
    CATEGORICAL_COLS,
    DROP_COLS,
    DROP_GENDER,
    TARGET_COL,
    PCA_VARIANCE_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
    MODELS_DIR,
    PIPELINE_FILENAME,
)


class DataPreprocessor:
    """
    Encapsula el pipeline de preprocesamiento completo.

    Attributes
    ----------
    label_encoders : dict[str, LabelEncoder]
        Encoders ajustados para cada variable categórica.
    scaler : StandardScaler
        Escalador ajustado sobre el conjunto de entrenamiento.
    pca : PCA
        Modelo PCA ajustado (n_components determinado por varianza).
    feature_columns : list[str]
        Nombres de las columnas de features en el orden de entrenamiento.
    n_pca_components : int
        Número de componentes PCA seleccionados.
    bmi_median : float
        Mediana del BMI usada para imputación.
    is_fitted : bool
        Indica si el preprocesador ya fue ajustado.
    """

    def __init__(self, variance_threshold: float = PCA_VARIANCE_THRESHOLD) -> None:
        self.variance_threshold = variance_threshold
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.feature_columns: list[str] = []
        self.n_pca_components: int = 0
        self.bmi_median: float = 0.0
        self.is_fitted: bool = False

    # ─── Métodos públicos ──────────────────────────────────────────────────────

    def fit_transform_full(
        self, data_path: str | Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga el CSV, limpia, ajusta todos los artefactos y devuelve los
        conjuntos listos para entrenar con SMOTE aplicado al train.

        Returns
        -------
        X_train_bal : np.ndarray
            Features de entrenamiento balanceadas (post-SMOTE).
        X_test_pca : np.ndarray
            Features de prueba transformadas (no balanceadas).
        y_train_bal : np.ndarray
            Etiquetas de entrenamiento balanceadas.
        y_test : np.ndarray
            Etiquetas de prueba originales.
        """
        df = self._load_and_clean(data_path)
        df_enc = self._encode_categoricals(df, fit=True)

        X = df_enc.drop(columns=[TARGET_COL])
        y = df_enc[TARGET_COL]
        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        X_train_pca, X_test_pca = self._fit_pca(X_train_scaled, X_test_scaled)

        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_bal, y_train_bal = smote.fit_resample(
            X_train_pca, y_train.reset_index(drop=True)
        )

        self.is_fitted = True
        return X_train_bal, X_test_pca, y_train_bal, y_test.to_numpy()

    def transform(self, patient_dict: dict) -> np.ndarray:
        """
        Transforma un único paciente (dict) al espacio PCA listo para
        ser enviado al clasificador.

        Parameters
        ----------
        patient_dict : dict
            Diccionario con los mismos campos que el dataset original
            (sin 'id' ni 'stroke').

        Returns
        -------
        np.ndarray con shape (1, n_pca_components)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "El preprocesador no ha sido ajustado. "
                "Llama a fit_transform_full() o carga un pipeline guardado."
            )

        row = pd.DataFrame([patient_dict])
        row = self._encode_categoricals(row, fit=False)
        row = row[self.feature_columns]

        row_scaled = self.scaler.transform(row)
        return self.pca.transform(row_scaled)

    def save(self, directory: Path = MODELS_DIR) -> Path:
        """Serializa todos los artefactos del pipeline en un único archivo."""
        path = directory / PIPELINE_FILENAME
        joblib.dump(
            {
                "label_encoders":    self.label_encoders,
                "scaler":            self.scaler,
                "pca":               self.pca,
                "feature_columns":   self.feature_columns,
                "n_pca_components":  self.n_pca_components,
                "bmi_median":        self.bmi_median,
                "variance_threshold": self.variance_threshold,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> "DataPreprocessor":
        """Carga un preprocesador previamente serializado."""
        path = directory / PIPELINE_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Pipeline no encontrado en: {path}")

        data = joblib.load(path)
        instance = cls(variance_threshold=data["variance_threshold"])
        instance.label_encoders   = data["label_encoders"]
        instance.scaler           = data["scaler"]
        instance.pca              = data["pca"]
        instance.feature_columns  = data["feature_columns"]
        instance.n_pca_components = data["n_pca_components"]
        instance.bmi_median       = data["bmi_median"]
        instance.is_fitted        = True
        return instance

    # ─── Métodos privados ──────────────────────────────────────────────────────

    def _load_and_clean(self, data_path: str | Path) -> pd.DataFrame:
        """Carga el CSV y aplica la limpieza básica del notebook."""
        df = pd.read_csv(data_path)

        # Eliminar columnas irrelevantes
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # Filtrar categorías raras en 'gender'
        df = df[~df["gender"].isin(DROP_GENDER)].copy()

        # Convertir BMI de string a numérico e imputar con mediana
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
        self.bmi_median = float(df["bmi"].median())
        df["bmi"] = df["bmi"].fillna(self.bmi_median)

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Aplica LabelEncoder a las columnas categóricas.
        Si fit=True ajusta y guarda los encoders; si False los reutiliza.
        """
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = le.transform(df[col])
        return df

    def _fit_pca(
        self,
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determina automáticamente el número de componentes PCA que
        captura el umbral de varianza requerido y ajusta el modelo.
        """
        pca_full = PCA(random_state=RANDOM_STATE)
        pca_full.fit(X_train_scaled)

        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        self.n_pca_components = int(np.argmax(cum_var >= self.variance_threshold) + 1)

        self.pca = PCA(n_components=self.n_pca_components, random_state=RANDOM_STATE)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca  = self.pca.transform(X_test_scaled)

        return X_train_pca, X_test_pca
