from typing import Any

import numpy as np
from sklearn import metrics

from .metric import Metric


class MaxError(Metric):
    def __call__(self, y_true: np.ndarray, y_predicted: np.ndarray) -> Any:
        return metrics.max_error(y_true, y_predicted)


class MeanAbsoluteError(Metric):
    def __call__(
        self,
        y_true: np.ndarray,
        y_predicted: np.ndarray,
    ) -> Any:
        return metrics.mean_absolute_error(
            y_true,
            y_predicted,
        )


class MeanSquaredError(Metric):
    def __call__(
        self,
        y_true: np.ndarray,
        y_predicted: np.ndarray,
    ) -> Any:
        return metrics.mean_squared_error(
            y_true,
            y_predicted,
        )


class R2Score(Metric):
    def __call__(
        self,
        y_true: np.ndarray,
        y_predicted: np.ndarray,
    ) -> Any:
        return metrics.r2_score(
            y_true,
            y_predicted,
        )
