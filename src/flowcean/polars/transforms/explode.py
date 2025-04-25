import logging
from collections.abc import Sequence

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

    def __init__(
        self,
        features: str | Sequence[str] | None = None,
        *more_features: str,
    ) -> None:
        self.features = features
        self.more_features = more_features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if self.features is None:
            self.features = data.columns
        return data.explode(self.features, *self.more_features)
