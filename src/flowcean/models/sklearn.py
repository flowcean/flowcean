from __future__ import annotations

import pickle
from io import BytesIO
from pathlib import Path
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
    def save(self, path: Path) -> None:
        model_bytes = BytesIO()
        joblib.dump(self.model, model_bytes)
        model_bytes.seek(0)
        data = {"data": model_bytes.read(), "output_name": self.output_name}
        with path.open("wb") as file:
            pickle.dump(data, file)

    @override
    @classmethod
    def load(cls, path: Path) -> SciKitModel:
        with path.open("rb") as file:
            data = pickle.load(file)  # noqa: S301
        return cls(joblib.load(BytesIO(data["data"])), data["output_name"])
