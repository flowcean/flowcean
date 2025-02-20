import logging
from collections.abc import Iterable

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class First(Transform):
    """Selects the first time value of a time-series feature."""

    def __init__(
        self,
        features: str | Iterable[str],
        *,
        replace: bool = False,
    ) -> None:
        """Initializes the First transform.

        Args:
            features: The features to apply this transform to.
            replace: Whether to replace the original features with the
                transformed ones. If set to False, the default, the value will
                be added as a new feature named `{feature}_first`.
        """
        self.features = [features] if isinstance(features, str) else features
        self.replace = replace

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        for feature in self.features:
            expr = (
                pl.col(feature)
                .list.eval(pl.element().struct.field("value"))
                .list.first()
            )
            data = data.with_columns(
                expr if self.replace else expr.alias(f"{feature}_first"),
            )
        return data
