import logging

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Rename(Transform):
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
