from pathlib import Path
from typing import Any

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # noqa: UP035

import joblib
import polars as pl

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
        joblib.dump(self.model, path)

    @override
    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
