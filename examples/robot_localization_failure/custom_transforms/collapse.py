import logging

import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class Collapse(Transform):
    def __init__(
        self,
        feature: str,
        element: int = 0,
    ) -> None:
        super().__init__()
        self.feature = feature
        self.element = element

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.with_columns(
            pl.col(self.feature)
            .list.get(self.element)
            .struct.field("value")
            .alias(self.feature),
        )
