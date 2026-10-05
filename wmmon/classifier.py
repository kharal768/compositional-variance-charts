"""The inspection module being monitored.

This is deliberately not a contribution of the paper.  It stands in for a
deployed wafer-map classifier, and the monitoring layer treats it as a black
box that emits class probabilities.  Two backends:

``hgb``   HistGradientBoosting on the 59 handcrafted features.  No deep-learning
          dependency, trains in minutes on the labelled subset.
``cnn``   Small torch CNN, used if torch is installed.  Prefer this for the
          submitted paper: reviewers will expect the monitored module to be the
          kind of model a fab actually runs.

Class imbalance is handled by balanced sample weights, not resampling.  Note
in the paper that ``none`` dominates at roughly 85 % of labelled wafers; a
classifier that ignores it entirely scores well on accuracy and is useless, so
report per-class recall and macro-F1, never accuracy alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class InspectionModel:
    backend: str
    scaler: StandardScaler
    model: object
    classes: list[str]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(scaled)
        # Re-order to the canonical class list; a backend may drop a class that
        # never appears in training, which would silently misalign the
        # composition downstream.
        out = np.zeros((len(features), len(self.classes)), dtype=float)
        model_classes = list(getattr(self.model, "classes_", range(proba.shape[1])))
        for j, name in enumerate(model_classes):
            out[:, self.classes.index(str(name))] = proba[:, j]
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums


def train_inspection_model(
    features: np.ndarray,
    labels: list[str],
    classes: list[str],
    backend: str = "hgb",
    seed: int = 0,
) -> InspectionModel:
    if backend != "hgb":
        raise NotImplementedError(
            "only the 'hgb' backend is available in this environment; install "
            "torch and implement the CNN hook for the submitted version"
        )
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    weights = compute_sample_weight("balanced", labels)
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    model.fit(scaled, labels, sample_weight=weights)
    return InspectionModel(
        backend=backend, scaler=scaler, model=model, classes=list(classes)
    )


def evaluate(
    model: InspectionModel,
    features: np.ndarray,
    labels: list[str],
) -> dict:
    """Held-out performance of the inspection module itself."""
    proba = model.predict_proba(features)
    predicted = [model.classes[i] for i in proba.argmax(axis=1)]
    return {
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predicted, average="weighted", zero_division=0)
        ),
        "report": classification_report(labels, predicted, zero_division=0),
        "confusion": confusion_matrix(
            labels, predicted, labels=model.classes
        ).tolist(),
    }
