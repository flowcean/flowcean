from __future__ import annotations

from io import BytesIO
from typing import Any

import joblib
import polars as pl
from typing_extensions import override

from flowcean.core import Model


class SciKitModel(Model):
    """A model that wraps a scikit-learn model."""

    def __init__(
        self,
        model: Any,
        output_names: list[str],
    ) -> None:
        """Initialize the model.

        Args:
            model: The scikit-learn model.
            output_names: The names of the output columns.
        """
        self.model = model
        self.output_names = output_names

    @override
    def predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        outputs = self.model.predict(input_features.collect())
        if len(self.output_names) == 1:
            data = {self.output_names[0]: outputs}
        else:
            data = {
                self.output_names[i]: outputs[:, i]
                for i in range(len(self.output_names))
            }
        return pl.DataFrame(data).lazy()

    @override
    def save_state(self) -> dict[str, Any]:
        model_bytes = BytesIO()
        joblib.dump(self.model, model_bytes)
        model_bytes.seek(0)
        return {
            "data": model_bytes.read(),
            "output_names": self.output_names,
        }

    @override
    @classmethod
    def load_from_state(cls, state: dict[str, Any]) -> SciKitModel:
        return cls(
            joblib.load(BytesIO(state["data"])),
            state["output_names"],
        )
