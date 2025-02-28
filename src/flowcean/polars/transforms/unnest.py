import logging
from collections.abc import Collection

import polars as pl
from polars._typing import ColumnNameOrSelector
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Unnest(Transform):
    """Decompose struct columns into separate columns for each field."""

    def __init__(
        self,
        features: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
    ) -> None:
        """Initializes the Unnest transform.

        Args:
            features: The features to unnest. Treats the selection as a
                parameter to polars `unnest` method. You can use regular
                expressions by wrapping the argument by ^ and $.
        """
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("selecting features %s", self.features)
        return data.unnest(self.features)
