from __future__ import annotations

from typing import Literal, override

import polars as pl
from sklearn import metrics

from flowcean.polars import PolarsMetric

from ._multioutput import MultiOutputMetric


class MaxError(PolarsMetric[float]):
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
    def _compute(self, true: pl.DataFrame, predicted: pl.DataFrame) -> float:
        return float(metrics.max_error(true, predicted))


class MeanAbsoluteError(MultiOutputMetric):
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
        true: pl.DataFrame,
        predicted: pl.DataFrame,
    ) -> float | dict[str, float]:
        error = metrics.mean_absolute_error(
            true,
            predicted,
            multioutput=self.multioutput,
        )
        return self._finalize_result(error, true)


class MeanAbsolutePercentageError(MultiOutputMetric):
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
        true: pl.DataFrame,
        predicted: pl.DataFrame,
    ) -> float | dict[str, float]:
        error = metrics.mean_absolute_percentage_error(
            true,
            predicted,
            multioutput=self.multioutput,
        )
        return self._finalize_result(error, true)


class MeanSquaredError(MultiOutputMetric):
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
        true: pl.DataFrame,
        predicted: pl.DataFrame,
    ) -> float | dict[str, float]:
        error = metrics.mean_squared_error(
            true,
            predicted,
            multioutput=self.multioutput,
        )
        return self._finalize_result(error, true)


class R2Score(MultiOutputMetric):
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
        true: pl.DataFrame,
        predicted: pl.DataFrame,
    ) -> float | dict[str, float]:
        error = metrics.r2_score(
            true,
            predicted,
            multioutput=self.multioutput,
        )
        return self._finalize_result(error, true)
