from typing import Any

import polars as pl
from sklearn import metrics
from typing_extensions import override

from flowcean.core import OfflineMetric
from flowcean.polars import LazyMixin, SelectMixin


class Accuracy(SelectMixin, LazyMixin, OfflineMetric):
    """Accuracy classification score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).
    """

    def __init__(
        self,
        features: list[str] | None = None,
    ) -> None:
        """Initialize metric.

        Args:
            features: The features to calculate the metric for. If None, the
                metric uses all features in the data.
        """
        super().__init__(features=features)

    @override
    def compute(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.accuracy_score(true, predicted)


class ClassificationReport(SelectMixin, LazyMixin, OfflineMetric):
    """Build a text report showing the main classification metrics.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
    """

    def __init__(
        self,
        features: list[str] | None = None,
    ) -> None:
        """Initialize metric.

        Args:
            features: The features to calculate the metric for. If None, the
                metric uses all features in the data.
        """
        super().__init__(features=features)

    @override
    def compute(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return "\n" + str(
            metrics.classification_report(true, predicted),
        )


class FBetaScore(SelectMixin, LazyMixin, OfflineMetric):
    """F-beta score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html).
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        features: list[str] | None = None,
    ) -> None:
        """Initialize metric.

        Args:
            beta: The beta parameter.
            features: The features to calculate the metric for. If None, the
                metric uses all features in the data.
        """
        super().__init__(features=features)
        self.beta = beta

    @override
    def compute(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.fbeta_score(true, predicted, beta=self.beta)


class PrecisionScore(SelectMixin, LazyMixin, OfflineMetric):
    """Precision classification score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html).
    """

    def __init__(
        self,
        features: list[str] | None = None,
    ) -> None:
        """Initialize metric.

        Args:
            features: The features to calculate the metric for. If None, the
                metric uses all features in the data.
        """
        super().__init__(features=features)

    @override
    def compute(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.precision_score(true, predicted)


class Recall(SelectMixin, LazyMixin, OfflineMetric):
    """Recall classification score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html).
    """

    def __init__(
        self,
        features: list[str] | None = None,
    ) -> None:
        """Initialize metric.

        Args:
            features: The features to calculate the metric for. If None, the
                metric uses all features in the data.
        """
        super().__init__(features=features)

    @override
    def compute(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        return metrics.recall_score(true, predicted)
