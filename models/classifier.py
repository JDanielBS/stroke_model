"""
StrokeClassifier
================
Gestiona el ciclo de vida completo del modelo de clasificación:

    - Entrenamiento de los 5 clasificadores (KNN, Decision Tree,
      Random Forest, AdaBoost, SVM).
    - Evaluación comparativa con métricas estándar.
    - Selección automática del mejor modelo por AUC-ROC.
    - Serialización (save) y deserialización (load) con joblib.
    - Predicción sobre nuevos pacientes.
"""

import json
import numpy as np
import joblib
from pathlib import Path
from typing import Any

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from stroke_api.config import (
    MODEL_PARAMS,
    RANDOM_STATE,
    MODELS_DIR,
    MODEL_FILENAME,
    METADATA_FILENAME,
)


# ─── Tipo alias ────────────────────────────────────────────────────────────────
Classifier = Any   # scikit-learn estimator compatible


class StrokeClassifier:
    """
    Entrena, evalúa, guarda y carga modelos de clasificación de stroke.

    Attributes
    ----------
    best_model_name : str
        Nombre del modelo con mejor AUC-ROC.
    best_model : Classifier
        Instancia del mejor clasificador entrenado.
    metrics : dict
        Diccionario con las métricas de todos los modelos entrenados.
    trained_models : dict[str, Classifier]
        Todos los modelos entrenados, indexados por nombre.
    is_trained : bool
        Indica si los modelos ya fueron entrenados.
    """

    def __init__(self) -> None:
        self.best_model_name: str = ""
        self.best_model: Classifier | None = None
        self.metrics: dict = {}
        self.trained_models: dict[str, Classifier] = {}
        self.is_trained: bool = False

    # ─── Entrenamiento ─────────────────────────────────────────────────────────

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Entrena los 5 clasificadores, los evalúa y selecciona el mejor.

        Parameters
        ----------
        X_train / y_train : datos de entrenamiento (post-SMOTE, post-PCA).
        X_test  / y_test  : datos de prueba (post-PCA, sin balanceo).

        Returns
        -------
        dict con las métricas comparativas de todos los modelos.
        """
        models: dict[str, Classifier] = {
            "KNN":            self._build_knn(X_train, y_train),
            "Árbol Decisión": self._build_decision_tree(X_train, y_train),
            "Random Forest":  self._build_random_forest(X_train, y_train),
            "AdaBoost":       self._build_adaboost(X_train, y_train),
            "SVM":            self._build_svm(X_train, y_train),
        }

        best_auc   = -1.0
        all_metrics: dict[str, dict] = {}

        for name, model in models.items():
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            auc     = roc_auc_score(y_test, y_proba)

            all_metrics[name] = {
                "accuracy":  round(float(accuracy_score(y_test, y_pred)),  4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall":    round(float(recall_score(y_test, y_pred)),    4),
                "f1_score":  round(float(f1_score(y_test, y_pred)),        4),
                "auc_roc":   round(float(auc),                             4),
                "classification_report": classification_report(
                    y_test, y_pred,
                    target_names=["Sin Stroke", "Con Stroke"]
                ),
            }

            if auc > best_auc:
                best_auc = auc
                self.best_model_name = name
                self.best_model      = model

        self.trained_models = models
        self.metrics        = all_metrics
        self.is_trained     = True
        return all_metrics

    # ─── Predicción ────────────────────────────────────────────────────────────

    def predict(self, X_pca: np.ndarray) -> dict:
        """
        Realiza la predicción sobre datos ya transformados (post-PCA).

        Returns
        -------
        dict con 'prediction' (int), 'probability' (float) y 'risk_level' (str).
        """
        if not self.is_trained or self.best_model is None:
            raise RuntimeError(
                "El clasificador no ha sido entrenado ni cargado. "
                "Llama a train_all() o usa load()."
            )

        pred  = int(self.best_model.predict(X_pca)[0])
        prob  = float(self.best_model.predict_proba(X_pca)[0][1])

        if prob >= 0.50:
            risk = "ALTO"
        elif prob >= 0.25:
            risk = "MODERADO"
        else:
            risk = "BAJO"

        return {
            "prediction":  pred,
            "probability": round(prob, 4),
            "risk_level":  risk,
            "model_used":  self.best_model_name,
        }

    # ─── Persistencia ──────────────────────────────────────────────────────────

    def save(self, directory: Path = MODELS_DIR) -> dict[str, Path]:
        """
        Guarda el mejor modelo y sus métricas en disco.

        Returns
        -------
        dict con las rutas de los archivos guardados.
        """
        if not self.is_trained:
            raise RuntimeError("Entrena el clasificador antes de guardarlo.")

        model_path    = directory / MODEL_FILENAME
        metadata_path = directory / METADATA_FILENAME

        joblib.dump(
            {
                "model":            self.best_model,
                "best_model_name":  self.best_model_name,
                "trained_models":   self.trained_models,
            },
            model_path,
        )

        with open(metadata_path, "w", encoding="utf-8") as f:
            # classification_report no es serializable directamente → lo conservamos como string
            serializable_metrics = {
                name: {k: v for k, v in vals.items()}
                for name, vals in self.metrics.items()
            }
            json.dump(
                {
                    "best_model_name": self.best_model_name,
                    "metrics":         serializable_metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return {"model": model_path, "metadata": metadata_path}

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> "StrokeClassifier":
        """
        Carga el mejor modelo serializado desde disco.

        Returns
        -------
        Instancia de StrokeClassifier lista para hacer predicciones.
        """
        model_path    = directory / MODEL_FILENAME
        metadata_path = directory / METADATA_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

        data = joblib.load(model_path)

        instance = cls()
        instance.best_model      = data["model"]
        instance.best_model_name = data["best_model_name"]
        instance.trained_models  = data.get("trained_models", {})
        instance.is_trained      = True

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            instance.metrics = meta.get("metrics", {})

        return instance

    def get_metrics_summary(self) -> dict:
        """Devuelve las métricas de todos los modelos (sin classification_report)."""
        return {
            name: {k: v for k, v in vals.items() if k != "classification_report"}
            for name, vals in self.metrics.items()
        }

    # ─── Constructores privados de cada modelo ─────────────────────────────────

    def _build_knn(self, X_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
        """Busca el mejor K por validación cruzada y entrena el modelo final."""
        params     = MODEL_PARAMS["knn"]
        k_values   = list(params["k_range"])
        auc_scores = []

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=params["metric"])
            scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="roc_auc")
            auc_scores.append(scores.mean())

        best_k = k_values[int(np.argmax(auc_scores))]
        model  = KNeighborsClassifier(n_neighbors=best_k, metric=params["metric"])
        model.fit(X_train, y_train)
        return model

    def _build_decision_tree(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> DecisionTreeClassifier:
        """Busca la profundidad óptima por validación cruzada y entrena el árbol."""
        params      = MODEL_PARAMS["decision_tree"]
        depths      = list(params["depth_range"])
        auc_val     = []

        for d in depths:
            dt     = DecisionTreeClassifier(max_depth=d, random_state=params["random_state"])
            scores = cross_val_score(dt, X_train, y_train, cv=5, scoring="roc_auc")
            auc_val.append(scores.mean())

        best_depth = depths[int(np.argmax(auc_val))]
        model = DecisionTreeClassifier(
            max_depth=best_depth,
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)
        return model

    def _build_random_forest(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> RandomForestClassifier:
        params = MODEL_PARAMS["random_forest"]
        model  = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            n_jobs=params["n_jobs"],
        )
        model.fit(X_train, y_train)
        return model

    def _build_adaboost(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> AdaBoostClassifier:
        params = MODEL_PARAMS["adaboost"]
        model  = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=params["base_max_depth"]),
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)
        return model

    def _build_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> SVC:
        params = MODEL_PARAMS["svm"]
        model  = SVC(
            kernel=params["kernel"],
            C=params["C"],
            probability=params["probability"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)
        return model
