import logging
from typing import override

import polars as pl

from flowcean.core import StatelessTransform

logger = logging.getLogger(__name__)


class Rename(StatelessTransform[pl.LazyFrame, pl.LazyFrame]):
    """Rename features in an environment."""

    def __init__(self, mapping: dict[str, str]) -> None:
        """Initializes the Rename transform.

        Args:
            mapping: Key value pairs that map from the old feature name to the
                new one.
        """
        self.mapping = mapping

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.rename(self.mapping)
