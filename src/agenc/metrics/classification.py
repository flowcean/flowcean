from typing import Any

import numpy as np
from sklearn import metrics

from .metric import Metric


class Accuracy(Metric):
    def __call__(self, y_true: np.ndarray, y_predicted: np.ndarray) -> Any:
        return metrics.accuracy_score(
            y_true,
            y_predicted,
        )


class ClassificationReport(Metric):
    def __call__(
        self,
        y_true: np.ndarray,
        y_predicted: np.ndarray,
    ) -> Any:
        return metrics.classification_report(
            y_true,
            y_predicted,
        )


class FBetaScore(Metric):
    def __init__(
        self,
        beta: float = 1.0,
    ) -> None:
        self.beta = beta

    def __call__(
        self,
        y_true: np.ndarray,
        y_predicted: np.ndarray,
    ) -> Any:
        return metrics.fbeta_score(
            y_true,
            y_predicted,
            beta=self.beta,
        )


class PrecisionScore(Metric):
    def __call__(
        self,
        y_true: np.ndarray,
        y_predicted: np.ndarray,
    ) -> Any:
        return metrics.precision_score(
            y_true,
            y_predicted,
        )


class Recall(Metric):
    def __call__(
        self,
        y_true: np.ndarray,
        y_predicted: np.ndarray,
    ) -> Any:
        return metrics.recall_score(
            y_true,
            y_predicted,
        )
