from __future__ import annotations

from io import BytesIO
from typing import Any

import joblib
import polars as pl
from typing_extensions import override

from flowcean.core.model import Model


class SciKitModel(Model):
    """A model that wraps a scikit-learn model."""

    def __init__(
        self,
        model: Any,
        output_name: str,
    ) -> None:
        """Initialize the model.

        Args:
            model: The scikit-learn model.
            output_name: The name of the output column.
        """
        self.model = model
        self.output_name = output_name

    @override
    def predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        outputs = self.model.predict(input_features.collect())
        return pl.DataFrame({self.output_name: outputs}).lazy()

    @override
    def save_state(self) -> dict[str, Any]:
        model_bytes = BytesIO()
        joblib.dump(self.model, model_bytes)
        model_bytes.seek(0)
        return {
            "data": model_bytes.read(),
            "output_name": self.output_name,
        }

    @override
    @classmethod
    def load_from_state(cls, state: dict[str, Any]) -> SciKitModel:
        return cls(
            joblib.load(BytesIO(state["data"])),
            state["output_name"],
        )
