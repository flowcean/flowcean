from collections.abc import Iterable

import polars as pl
from typing_extensions import override

from flowcean.core import Transform


class Drop(Transform):
    """Drop features from the data."""

    def __init__(self, features: str | Iterable[str]) -> None:
        """Initializes the Drop transform."""
        super().__init__()
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.drop(self.features)
