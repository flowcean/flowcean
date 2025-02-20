import logging
from collections.abc import Iterable

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Mean(Transform):
    """Replaces time-series features with their mean value."""

    def __init__(
        self,
        features: str | Iterable[str],
        *,
        replace: bool = False,
    ) -> None:
        """Initializes the Mean transform.

        Args:
            features: The feature or features the mean should be calculated
                for.
            replace: Whether to replace the original features with the
                transformed ones. If set to False, the default, the value will
                be added as a new feature named `{feature}_mean`.
        """
        self.features = [features] if isinstance(features, str) else features
        self.replace = replace

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        for feature in self.features:
            expr = (
                pl.col(feature)
                .list.eval(pl.element().struct.field("value"))
                .list.mean()
            )
            data = data.with_columns(
                expr if self.replace else expr.alias(f"{feature}_mean"),
            )

        return data
