import logging

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Explode(Transform):
    """This wraps the `explode` method of Polars.

    Reference: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.explode.html#polars.DataFrame.explode
    If `features` is None, all columns will be exploded.

    Args:
        features (list[str] | None): List of features to explode.

    """

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Exploding timeseries")
        if self.features is None:
            # explode all columns
            self.features = data.columns
        return data.explode(self.features)
