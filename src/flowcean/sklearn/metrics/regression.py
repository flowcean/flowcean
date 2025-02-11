from typing import Any

import polars as pl
from sklearn import metrics
from typing_extensions import override

from flowcean.core import OfflineMetric


class MaxError(OfflineMetric):
    """Max error regression loss.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html).
    """

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.max_error(
            true.collect(streaming=True),
            predicted.collect(streaming=True),
        )


class MeanAbsoluteError(OfflineMetric):
    """Mean absolute error (MAE) regression loss.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).
    """

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.mean_absolute_error(
            true.collect(streaming=True),
            predicted.collect(streaming=True),
        )


class MeanSquaredError(OfflineMetric):
    """Mean squared error (MSE) regression loss.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html).
    """

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.mean_squared_error(
            true.collect(streaming=True),
            predicted.collect(streaming=True),
        )


class R2Score(OfflineMetric):
    """R^2 (coefficient of determination) regression score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).
    """

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.r2_score(
            true.collect(streaming=True),
            predicted.collect(streaming=True),
        )
