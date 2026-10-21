from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol, override, runtime_checkable

import numpy as np
import polars as pl

from flowcean.polars.model import PolarsModel

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


@runtime_checkable
class SupportsPredict(Protocol):
    """The prediction boundary of a scikit-learn estimator."""

    def predict(self, X: Any) -> object: ...


@runtime_checkable
class SupportsClassifier(SupportsPredict, Protocol):
    """A fitted classifier exposing labels and class probabilities."""

    @property
    def classes_(self) -> object: ...

    def predict_proba(self, X: Any) -> object: ...


def prediction_frame(
    outputs: object,
    rows: int,
    names: list[str],
) -> pl.LazyFrame:
    """Validate estimator output before assigning target column names."""
    if not isinstance(outputs, np.ndarray):
        message = "Estimator predictions must be NumPy arrays"
        raise TypeError(message)
    if outputs.ndim == 1 and len(names) == 1:
        outputs = outputs.reshape(-1, 1)
    if outputs.ndim != 2 or outputs.shape != (rows, len(names)):
        message = "Prediction shape must match input rows and output columns"
        raise ValueError(message)
    return pl.DataFrame(outputs, schema=names, orient="row").lazy()


def validate_threshold(threshold: float | None) -> None:
    """Require an optional finite probability threshold."""
    if threshold is not None and (
        not math.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        message = "threshold must be finite and between 0 and 1"
        raise ValueError(message)


def binary_classes(
    classes: object,
    names: list[str],
) -> NDArray[Any]:
    """Validate the single-target binary probability contract."""
    if (
        len(names) != 1
        or not isinstance(classes, np.ndarray)
        or classes.ndim != 1
        or len(classes) != 2
    ):
        message = "Binary probabilities require one target and two classes"
        raise ValueError(message)
    return classes


def positive_probabilities(
    probabilities: object,
    rows: int,
) -> NDArray[Any]:
    """Validate the binary probability matrix and select classes_[1]."""
    if (
        not isinstance(probabilities, np.ndarray)
        or probabilities.ndim != 2
        or probabilities.shape != (rows, 2)
    ):
        message = "Binary probabilities must have shape (input rows, 2)"
        raise ValueError(message)
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        message = "Probabilities must be finite and between 0 and 1"
        raise ValueError(message)
    return probabilities[:, 1]


class SciKitModel(PolarsModel):
    """A model that wraps a scikit-learn estimator."""

    def __init__(
        self,
        estimator: SupportsPredict,
        *,
        output_names: Iterable[str],
        name: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(estimator, SupportsPredict) or not callable(
            estimator.predict
        ):
            message = "Estimator must implement predict"
            raise TypeError(message)
        self._name = name or estimator.__class__.__name__
        self.estimator = estimator
        self.output_names = list(output_names)

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()
        return prediction_frame(
            self.estimator.predict(input_features),
            input_features.height,
            self.output_names,
        )


class SciKitClassifierModel(SciKitModel):
    """Native classifier predictions, or explicit binary thresholding.

    ``None`` preserves the estimator's labels and multiclass predictions.
    ``predict_proba`` exposes only the probability of ``classes_[1]`` for
    single-target binary classifiers, without label postprocessing.
    """

    def __init__(
        self,
        estimator: SupportsClassifier,
        *,
        output_names: Iterable[str],
        threshold: float | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(estimator, SupportsClassifier) or not callable(
            estimator.predict_proba
        ):
            message = (
                "Classifier must expose predict, predict_proba and classes_"
            )
            raise TypeError(message)
        super().__init__(estimator, output_names=output_names, name=name)
        self.classifier = estimator
        validate_threshold(threshold)
        self.threshold = threshold
        if threshold is not None:
            binary_classes(self.classifier.classes_, self.output_names)

    def _predict_proba(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()
        binary_classes(self.classifier.classes_, self.output_names)
        probabilities = positive_probabilities(
            self.classifier.predict_proba(input_features),
            input_features.height,
        )
        return prediction_frame(
            probabilities,
            input_features.height,
            self.output_names,
        )

    def predict_proba(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Predict binary probabilities, applying preprocessing exactly once."""
        return self._predict_proba(self.preprocess(input_features))

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        validate_threshold(self.threshold)
        if self.threshold is None:
            return super()._predict(input_features)
        classes = binary_classes(self.classifier.classes_, self.output_names)
        probabilities = self._predict_proba(input_features).collect()
        labels = np.where(
            probabilities.to_series().to_numpy() >= self.threshold,
            classes[1],
            classes[0],
        )
        return prediction_frame(
            labels, probabilities.height, self.output_names
        )
