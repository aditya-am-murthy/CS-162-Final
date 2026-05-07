"""Simple mislabeled-example detector from cartography metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


@dataclass
class ConstantNoiseModel:
    label: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(x),), fill_value=self.label, dtype=int)


def build_feature_matrix(rows: List[Dict]) -> np.ndarray:
    features = []
    for r in rows:
        features.append(
            [
                float(r["confidence"]),
                float(r["variability"]),
                float(r.get("correctness", 0.0)),
            ]
        )
    return np.array(features, dtype=float)


def train_noise_detector(
    train_rows: List[Dict], noisy_flags: List[int]
) -> Union[LogisticRegression, ConstantNoiseModel]:
    x = build_feature_matrix(train_rows)
    y = np.array(noisy_flags, dtype=int)
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return ConstantNoiseModel(label=int(unique_labels[0]))
    model = LogisticRegression(max_iter=200)
    model.fit(x, y)
    return model


def evaluate_noise_detector(
    model: Union[LogisticRegression, ConstantNoiseModel],
    rows: List[Dict],
    noisy_flags: List[int],
) -> Tuple[float, float, float]:
    x = build_feature_matrix(rows)
    y_true = np.array(noisy_flags, dtype=int)
    y_pred = model.predict(x)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(precision), float(recall), float(f1)

