from collections.abc import Iterable
from typing import override

import polars as pl

from flowcean.core.transform import Transform


class Drop(Transform):
    """Drop features from the data."""

    def __init__(self, features: str | Iterable[str]) -> None:
        """Initializes the Drop transform."""
        super().__init__()
        self.features = features

    @override
    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.drop(self.features)
