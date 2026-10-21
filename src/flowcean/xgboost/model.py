from copy import copy
from typing import Any, override

import numpy as np
import polars as pl

from flowcean._optional import raise_for_missing_optional_dependency

try:
    from xgboost import XGBClassifier, XGBRegressor
except ModuleNotFoundError as error:
    raise_for_missing_optional_dependency(
        error,
        extra="xgboost",
        module="flowcean.xgboost.model",
        missing_dependencies={"xgboost"},
    )

from flowcean.core.model import Model
from flowcean.sklearn.model import (
    binary_classes,
    positive_probabilities,
    prediction_frame,
    validate_threshold,
)


class XGBoostClassifierModel(Model[pl.LazyFrame, pl.LazyFrame]):
    """Wrapper for an XGBoost classifier model with threshold support."""

    classifier: XGBClassifier

    input_features: list[str]
    output_features: list[str]

    def __init__(
        self,
        classifier: XGBClassifier,
        *,
        input_features: list[str],
        output_features: list[str],
        threshold: float | None = None,
    ) -> None:
        super().__init__()
        validate_threshold(threshold)
        if threshold is not None:
            binary_classes(classifier.classes_, output_features)
        self.classifier = classifier
        self.input_features = input_features
        self.output_features = output_features
        self.threshold = threshold

    def _predict_proba(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Predict probability of positive class."""
        binary_classes(self.classifier.classes_, self.output_features)
        inputs = input_features.select(self.input_features).collect()
        proba = positive_probabilities(
            self.classifier.predict_proba(inputs.to_numpy()),
            inputs.height,
        )
        return prediction_frame(proba, inputs.height, self.output_features)

    def predict_proba(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        """Predict class probabilities, applying preprocessing transforms.

        Args:
            input_features: The inputs for which to predict probabilities.

        Returns:
            The predicted probabilities for the positive class.
        """
        input_features = self.preprocess(input_features)
        return self._predict_proba(input_features)

    def __getstate__(self) -> dict[str, Any]:
        """Remove callbacks from a copy, preserving the live fitted estimator."""
        state = self.__dict__.copy()
        classifier = copy(self.classifier)
        classifier.callbacks = None
        state["classifier"] = classifier
        return state

    @override
    def _predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Preserve native predictions unless a binary threshold is explicit."""
        validate_threshold(self.threshold)
        if self.threshold is not None:
            classes = binary_classes(
                self.classifier.classes_, self.output_features
            )
            probas = self._predict_proba(input_features).collect()
            labels = np.where(
                probas.to_series().to_numpy() >= self.threshold,
                classes[1],
                classes[0],
            )
            return prediction_frame(
                labels, probas.height, self.output_features
            )

        inputs = input_features.select(self.input_features).collect()
        return prediction_frame(
            self.classifier.predict(inputs.to_numpy()),
            inputs.height,
            self.output_features,
        )


class XGBoostRegressorModel(Model[pl.LazyFrame, pl.LazyFrame]):
    """Wrapper for an XGBoost regressor model."""

    regressor: XGBRegressor

    input_features: list[str]
    output_features: list[str]

    def __init__(
        self,
        regressor: XGBRegressor,
        *,
        input_features: list[str],
        output_features: list[str],
    ) -> None:
        super().__init__()
        self.regressor = regressor
        self.input_features = input_features
        self.output_features = output_features

    def __getstate__(self) -> dict[str, Any]:
        """Remove callbacks from a copy, preserving the live fitted estimator."""
        state = self.__dict__.copy()
        regressor = copy(self.regressor)
        regressor.callbacks = None
        state["regressor"] = regressor
        return state

    @override
    def _predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        inputs = input_features.select(self.input_features).collect()
        return prediction_frame(
            self.regressor.predict(inputs.to_numpy()),
            inputs.height,
            self.output_features,
        )
