from typing import Any

from numpy.typing import NDArray
from sklearn import metrics

from agenc.core import Metric


class MaxError(Metric):
    def __call__(self, y_true: NDArray[Any], y_predicted: NDArray[Any]) -> Any:
        return metrics.max_error(y_true, y_predicted)


class MeanAbsoluteError(Metric):
    """Mean absolute error (MAE) regression loss.

    The :class:`MeanAbsoluteError` metric computes the MAE as defined by
    `scikit-learn
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html>`.
    """

    def __call__(
        self,
        y_true: NDArray[Any],
        y_predicted: NDArray[Any],
    ) -> Any:
        return metrics.mean_absolute_error(
            y_true,
            y_predicted,
        )


class MeanSquaredError(Metric):
    def __call__(
        self,
        y_true: NDArray[Any],
        y_predicted: NDArray[Any],
    ) -> Any:
        return metrics.mean_squared_error(
            y_true,
            y_predicted,
        )


class R2Score(Metric):
    def __call__(
        self,
        y_true: NDArray[Any],
        y_predicted: NDArray[Any],
    ) -> Any:
        return metrics.r2_score(
            y_true,
            y_predicted,
        )
