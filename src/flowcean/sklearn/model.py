from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import polars as pl
from typing_extensions import override

from flowcean.core.model import Model

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


class SupportsPredict(Protocol):
    """Protocol describing an object that has a `predict` method."""

    def predict(self, X: Any) -> NDArray: ...  # noqa: N803


class SupportsPredictProba(Protocol):
    """Protocol describing an object that has a `predict_proba` method."""

    def predict_proba(self, X: Any) -> NDArray: ...  # noqa: N803


class SciKitModel(Model):
    """A model that wraps a scikit-learn model.

    For classifiers with predict_proba, this model supports threshold-based
    predictions. Set the threshold attribute to customize the decision boundary.
    """

    estimator: SupportsPredict
    output_names: list[str]

    def __init__(
        self,
        estimator: SupportsPredict,
        *,
        output_names: Iterable[str],
        threshold: float | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            estimator: The scikit-learn estimator.
            output_names: The names of the output columns.
            threshold: Optional decision threshold for classifiers (default: None).
                If set and estimator has predict_proba, uses threshold-based
                prediction. Otherwise uses estimator's default predict method.
            name: The name of the model.
        """
        super().__init__()
        if name is None:
            name = estimator.__class__.__name__
        self._name = name
        self.estimator = estimator
        self.output_names = list(output_names)
        self.threshold = threshold

    def _predict_proba(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Predict class probabilities (only for classifiers)."""
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()

        if not hasattr(self.estimator, "predict_proba"):
            msg = (
                f"Estimator {self.estimator.__class__.__name__} "
                "does not support predict_proba"
            )
            raise AttributeError(msg)

        probas = self.estimator.predict_proba(input_features)[:, 1]

        if len(self.output_names) == 1:
            data = {self.output_names[0]: probas}
        else:
            data = {
                self.output_names[i]: probas[:, i]
                for i in range(len(self.output_names))
            }
        return pl.LazyFrame(data)

    def predict_proba(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Predict class probabilities, applying preprocessing transforms.

        Only available for classifiers that implement predict_proba.

        Args:
            input_features: The inputs for which to predict probabilities.

        Returns:
            The predicted probabilities for the positive class.
        """
        input_features = self.preprocess(input_features)
        return self._predict_proba(input_features)

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()

        # Use threshold-based prediction if threshold is set and model supports it
        if (
            self.threshold is not None
            and hasattr(self.estimator, "predict_proba")
        ):
            probas = self._predict_proba(input_features).collect()
            predictions = {}
            for col in probas.columns:
                predictions[col] = (probas[col] >= self.threshold).cast(
                    pl.Int64,
                )
            return pl.LazyFrame(predictions)

        # Otherwise use default predict
        outputs = self.estimator.predict(input_features)
        if len(self.output_names) == 1:
            data = {self.output_names[0]: outputs}
        else:
            data = {
                self.output_names[i]: outputs[:, i]
                for i in range(len(self.output_names))
            }
        return pl.LazyFrame(data)
