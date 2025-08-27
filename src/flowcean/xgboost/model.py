from typing import Any, cast

import polars as pl
from typing_extensions import override
from xgboost import Booster, XGBClassifier, XGBRegressor

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
    def predict(
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

    @override
    def save_state(self) -> dict[str, Any]:
        booster_name = self.classifier.booster
        model_bytes = self.classifier.get_booster().save_raw()
        return {
            "booster": booster_name,
            "data": model_bytes,
            "input_features": self.input_features,
            "output_features": self.output_features,
        }

    @override
    @classmethod
    def load_from_state(
        cls,
        state: dict[str, Any],
    ) -> "XGBoostClassifierModel":
        classifier = XGBClassifier(
            booster=state["booster"],
        )
        classifier._Booster = Booster()  # noqa: SLF001
        classifier.get_booster().load_model(cast("bytearray", state["data"]))

        return cls(
            classifier,
            input_features=state["input_features"],
            output_features=state["output_features"],
        )


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
    def predict(
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

    @override
    def save_state(self) -> dict[str, Any]:
        booster_name = self.regressor.booster
        model_bytes = self.regressor.get_booster().save_raw()
        return {
            "booster": booster_name,
            "data": model_bytes,
            "input_features": self.input_features,
            "output_features": self.output_features,
        }

    @override
    @classmethod
    def load_from_state(
        cls,
        state: dict[str, Any],
    ) -> "XGBoostRegressorModel":
        classifier = XGBRegressor(
            booster=state["booster"],
        )
        classifier._Booster = Booster()  # noqa: SLF001
        classifier.get_booster().load_model(cast("bytearray", state["data"]))

        return cls(
            classifier,
            input_features=state["input_features"],
            output_features=state["output_features"],
        )
