import logging
from collections.abc import Iterable

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class First(Transform):
    """Selects the first time value of a time-series feature."""

    def __init__(self, features: str | Iterable[str]) -> None:
        """Initializes the First transform.

        Args:
            features: The features to apply this transform to.
        """
        self.features = [features] if isinstance(features, str) else features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        for feature in self.features:
            data = data.with_columns(
                pl.col(feature)
                .list.eval(pl.element().struct.field("value"))
                .list.first()
                .alias(f"{feature}_first"),
            )
        return data
