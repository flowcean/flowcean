from typing import override

import polars as pl
from sklearn import metrics

from flowcean.polars import PolarsMetric


class Accuracy(PolarsMetric[float]):
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
    def _compute(self, true: pl.DataFrame, predicted: pl.DataFrame) -> float:
        return float(metrics.accuracy_score(true, predicted))


class ClassificationReport(PolarsMetric[str]):
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
    def _compute(self, true: pl.DataFrame, predicted: pl.DataFrame) -> str:
        return "\n" + str(
            metrics.classification_report(true, predicted),
        )


class FBetaScore(PolarsMetric[float]):
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
    def _compute(self, true: pl.DataFrame, predicted: pl.DataFrame) -> float:
        return float(metrics.fbeta_score(true, predicted, beta=self.beta))


class PrecisionScore(PolarsMetric[float]):
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
    def _compute(self, true: pl.DataFrame, predicted: pl.DataFrame) -> float:
        return float(metrics.precision_score(true, predicted))


class Recall(PolarsMetric[float]):
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
    def _compute(self, true: pl.DataFrame, predicted: pl.DataFrame) -> float:
        return float(metrics.recall_score(true, predicted))
