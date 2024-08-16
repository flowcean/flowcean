import logging
from typing import override

import polars as pl

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
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.rename(self.mapping)
