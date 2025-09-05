from typing import Any

from sklearn import metrics
from typing_extensions import override

from flowcean.core import Data, Metric
from flowcean.polars import LazyMixin, SelectMixin


class Accuracy(SelectMixin, LazyMixin, Metric):
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
    def _compute(self, true: Data, predicted: Data) -> Any:
        return metrics.accuracy_score(true, predicted)


class ClassificationReport(SelectMixin, LazyMixin, Metric):
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
    def _compute(self, true: Data, predicted: Data) -> Any:
        return "\n" + str(
            metrics.classification_report(true, predicted),
        )


class FBetaScore(SelectMixin, LazyMixin, Metric):
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
    def _compute(self, true: Data, predicted: Data) -> Any:
        return metrics.fbeta_score(true, predicted, beta=self.beta)


class PrecisionScore(SelectMixin, LazyMixin, Metric):
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
    def _compute(self, true: Data, predicted: Data) -> Any:
        return metrics.precision_score(true, predicted)


class Recall(SelectMixin, LazyMixin, Metric):
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
    def _compute(self, true: Data, predicted: Data) -> Any:
        return metrics.recall_score(true, predicted)
