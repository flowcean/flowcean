import logging
from collections.abc import Iterable
from typing import override

import polars as pl
from polars.type_aliases import IntoExpr

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Unnest(Transform):
    """Decompose struct columns into separate columns for each of their fields."""

    def __init__(self, features: IntoExpr | Iterable[IntoExpr]) -> None:
        """Initializes the Unnest transform.

        Args:
            features: The features to select. Treats the selection as a
                parameter to polars `select` method. You can use regular
                expressions by wrapping the argument by ^ and $.
        """
        self.features = features

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        logger.debug("selecting features %s", self.features)
        return data.unnest(self.features)
