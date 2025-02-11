import logging
from collections.abc import Iterable

import polars as pl
from polars._typing import IntoExpr
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Select(Transform):
    """Selects a subset of features from the data."""

    def __init__(self, features: IntoExpr | Iterable[IntoExpr]) -> None:
        """Initializes the Select transform.

        Args:
            features: The features to select. Treats the selection as a
                parameter to polars `select` method. You can use regular
                expressions by wrapping the argument by ^ and $.
        """
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("selecting features %s", self.features)
        return data.select(self.features)
