from typing import Any

import polars as pl
from sklearn import metrics
from typing_extensions import override

from flowcean.core import OfflineMetric


class Accuracy(OfflineMetric):
    """Accuracy classification score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).
    """

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.accuracy_score(
            true.collect(engine="streaming"),
            predicted.collect(engine="streaming"),
        )


class ClassificationReport(OfflineMetric):
    """Build a text report showing the main classification metrics.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
    """

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.classification_report(
            true.collect(engine="streaming"),
            predicted.collect(engine="streaming"),
        )


class FBetaScore(OfflineMetric):
    """F-beta score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html).
    """

    def __init__(self, beta: float = 1.0) -> None:
        """Initialize the metric.

        Args:
            beta: The beta parameter.
        """
        self.beta = beta

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.fbeta_score(
            true.collect(engine="streaming"),
            predicted.collect(engine="streaming"),
            beta=self.beta,
        )


class PrecisionScore(OfflineMetric):
    """Precision classification score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html).
    """

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.precision_score(
            true.collect(engine="streaming"),
            predicted.collect(engine="streaming"),
        )


class Recall(OfflineMetric):
    """Recall classification score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html).
    """

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.recall_score(
            true.collect(engine="streaming"),
            predicted.collect(engine="streaming"),
        )
