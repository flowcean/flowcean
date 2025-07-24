from typing import Any, cast

import polars as pl
from typing_extensions import override
from xgboost import Booster, XGBClassifier

from flowcean.core.model import Model


class XGBoostModel(Model):
    """Wrapper for an XGBoost model."""

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
    def load_from_state(cls, state: dict[str, Any]) -> "XGBoostModel":
        classifier = XGBClassifier(
            booster=state["booster"],
        )
        classifier._Booster = Booster({"n_jobs": 1})
        classifier.get_booster().load_model(cast("bytearray", state["data"]))

        return cls(
            classifier,
            input_features=state["input_features"],
            output_features=state["output_features"],
        )
