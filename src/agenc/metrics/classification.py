from typing import Any

import polars as pl
from sklearn import metrics
from typing_extensions import override

from agenc.core import Metric


class Accuracy(Metric):
    """Accuracy classification score.

    As defined by `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`.
    """

    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        return metrics.accuracy_score(true, predicted)


class ClassificationReport(Metric):
    """Build a text report showing the main classification metrics.

    As defined by `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html>`.
    """

    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        return metrics.classification_report(true, predicted)


class FBetaScore(Metric):
    """F-beta score.

    As defined by `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html>`.
    """

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        return metrics.fbeta_score(true, predicted, beta=self.beta)


class PrecisionScore(Metric):
    """Precision classification score.

    As defined by `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`.
    """

    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        return metrics.precision_score(true, predicted)


class Recall(Metric):
    """Recall classification score.

    As defined by `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html>`.
    """

    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        return metrics.recall_score(true, predicted)
