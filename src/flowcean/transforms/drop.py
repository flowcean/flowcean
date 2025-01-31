from collections.abc import Iterable

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # noqa: UP035

import polars as pl

from flowcean.core.transform import Transform


class Drop(Transform):
    """Drop features from the data."""

    def __init__(self, features: str | Iterable[str]) -> None:
        """Initializes the Drop transform."""
        super().__init__()
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.drop(self.features)
