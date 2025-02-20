import logging
from collections.abc import Iterable

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Mean(Transform):
    """Replaces time-series features with there mean value."""

    def __init__(self, features: str | Iterable[str]) -> None:
        """Initializes the Mean transform.

        Args:
            features: The feature or features the mean should be calculated
                for.
        """
        self.features = [features] if isinstance(features, str) else features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        for feature in self.features:
            data = data.with_columns(
                pl.col(feature)
                .list.eval(pl.element().struct.field("value"))
                .list.mean()
                .alias(f"{feature}_mean"),
            )

        return data
