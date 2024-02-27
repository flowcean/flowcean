import logging
from collections.abc import Iterable

import polars as pl
from polars.type_aliases import IntoExpr

from agenc.core import Transform

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

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        logger.debug(f"selecting features {self.features}")
        return data.select(self.features)
