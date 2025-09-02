import polars as pl
from typing_extensions import override
from xgboost import XGBClassifier, XGBRegressor

from flowcean.core.model import Model


class XGBoostClassifierModel(Model):
    """Wrapper for an XGBoost classifier model."""

    classifier: XGBClassifier

    input_features: list[str]
    output_features: list[str]

    def __init__(
        self,
        classifier: XGBClassifier,
        *,
        input_features: list[str],
        output_features: list[str],
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.input_features = input_features
        self.output_features = output_features

    @override
    def _predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        return pl.from_numpy(
            self.classifier.predict(
                input_features.select(self.input_features)
                .collect()
                .to_numpy(),
            ),
            self.output_features,
        ).lazy()


class XGBoostRegressorModel(Model):
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

    @override
    def _predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        return pl.from_numpy(
            self.regressor.predict(
                input_features.select(self.input_features)
                .collect()
                .to_numpy(),
            ),
            self.output_features,
        ).lazy()
