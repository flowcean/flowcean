from typing import Any

import polars as pl
from sklearn import metrics
from typing_extensions import override

from flowcean.core import Metric


class MaxError(Metric):
    """Max error regression loss.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html).
    """

    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        return metrics.max_error(true, predicted)


class MeanAbsoluteError(Metric):
    """Mean absolute error (MAE) regression loss.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).
    """

    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        return metrics.mean_absolute_error(true, predicted)


class MeanSquaredError(Metric):
    """Mean squared error (MSE) regression loss.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html).
    """

    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        return metrics.mean_squared_error(true, predicted)


class R2Score(Metric):
    """R^2 (coefficient of determination) regression score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).
    """

    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        return metrics.r2_score(true, predicted)
