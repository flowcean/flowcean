from pathlib import Path
from typing import Any

import joblib
import polars as pl

from flowcean.core.model import Model


class SciKitModel(Model):
    def __init__(
        self,
        model: Any,
        output_name: str,
    ) -> None:
        self.model = model
        self.output_name = output_name

    def predict(
        self,
        input_features: pl.DataFrame,
    ) -> pl.DataFrame:
        outputs = self.model.predict(input_features)
        return pl.DataFrame({self.output_name: outputs})

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
