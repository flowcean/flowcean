from __future__ import annotations

from typing import Literal

from sklearn import metrics
from typing_extensions import override

from flowcean.core import Data, Metric, Reportable
from flowcean.polars import LazyMixin, SelectMixin
from flowcean.sklearn.metrics import MultiOutputMixin


class MaxError(SelectMixin, LazyMixin, Metric):
    """Max error regression loss.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html).
    """

    def __init__(self, feature: str | None = None) -> None:
        """Initialize MaxError metric.

        Args:
            feature: The feature to calculate the metric for. If None, the
                metric expects a single feature in the data.
        """
        features = [feature] if feature is not None else None
        super().__init__(features=features)

    @override
    def _compute(self, true: Data, predicted: Data) -> Reportable:
        return metrics.max_error(true, predicted)


class MeanAbsoluteError(
    SelectMixin,
    LazyMixin,
    MultiOutputMixin,
    Metric,
):
    """Mean absolute error (MAE) regression loss.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).
    """

    def __init__(
        self,
        features: list[str] | None = None,
        multioutput: Literal[
            "raw_values",
            "uniform_average",
        ] = "raw_values",
    ) -> None:
        """Initialize metric.

        Args:
            features: The features to calculate the metric for. If None, the
                metric uses all features in the data.
            multioutput: Defines how to aggregate multiple output values.
                See [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
                for details.
        """
        super().__init__(features=features, multioutput=multioutput)

    @override
    def _compute(
        self,
        true: Data,
        predicted: Data,
    ) -> Reportable | dict[str, Reportable]:
        error = metrics.mean_absolute_error(
            true,
            predicted,
            multioutput=self.multioutput,
        )
        return self._finalize_result(error, true)


class MeanAbsolutePercentageError(
    SelectMixin,
    LazyMixin,
    MultiOutputMixin,
    Metric,
):
    """Mean absolute percentage error (MAPE) regression loss.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html).
    """

    def __init__(
        self,
        features: list[str] | None = None,
        multioutput: Literal[
            "raw_values",
            "uniform_average",
        ] = "raw_values",
    ) -> None:
        """Initialize metric.

        Args:
            features: The features to calculate the metric for. If None, the
                metric uses all features in the data.
            multioutput: Defines how to aggregate multiple output values.
                See [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
                for details.
        """
        super().__init__(features=features, multioutput=multioutput)

    @override
    def _compute(
        self,
        true: Data,
        predicted: Data,
    ) -> Reportable | dict[str, Reportable]:
        error = metrics.mean_absolute_percentage_error(
            true,
            predicted,
            multioutput=self.multioutput,
        )
        return self._finalize_result(error, true)


class MeanSquaredError(
    SelectMixin,
    LazyMixin,
    MultiOutputMixin,
    Metric,
):
    """Mean squared error (MSE) regression loss.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html).
    """

    def __init__(
        self,
        features: list[str] | None = None,
        multioutput: Literal[
            "raw_values",
            "uniform_average",
        ] = "raw_values",
    ) -> None:
        """Initialize metric.

        Args:
            features: The features to calculate the metric for. If None, the
                metric uses all features in the data.
            multioutput: Defines how to aggregate multiple output values.
                See [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
                for details.
        """
        super().__init__(features=features, multioutput=multioutput)

    @override
    def _compute(
        self,
        true: Data,
        predicted: Data,
    ) -> Reportable | dict[str, Reportable]:
        error = metrics.mean_squared_error(
            true,
            predicted,
            multioutput=self.multioutput,
        )
        return self._finalize_result(error, true)


class R2Score(
    SelectMixin,
    LazyMixin,
    MultiOutputMixin,
    Metric,
):
    """R^2 (coefficient of determination) regression score.

    As defined by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).
    """

    def __init__(
        self,
        features: list[str] | None = None,
        multioutput: Literal[
            "raw_values",
            "uniform_average",
        ] = "raw_values",
    ) -> None:
        """Initialize metric.

        Args:
            features: The features to calculate the metric for. If None, the
                metric uses all features in the data.
            multioutput: Defines how to aggregate multiple output values.
                See [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
                for details.
        """
        super().__init__(features=features, multioutput=multioutput)

    @override
    def _compute(
        self,
        true: Data,
        predicted: Data,
    ) -> Reportable | dict[str, Reportable]:
        error = metrics.r2_score(
            true,
            predicted,
            multioutput=self.multioutput,
        )
        return self._finalize_result(error, true)
