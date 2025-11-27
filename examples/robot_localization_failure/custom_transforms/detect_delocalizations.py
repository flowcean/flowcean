import logging

import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class DetectDelocalizations(Transform):
    def __init__(
        self,
        counter_column: str,
        name: str = "slice_points",
    ) -> None:
        super().__init__()
        self.counter_column = counter_column
        self.name = name

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug(
            "Extracting delocalization timestamps from column '%s' → '%s'",
            self.counter_column,
            self.name,
        )
        if self.counter_column in data.columns:
            # Simply extract the "time" field from each struct in the list
            return data.with_columns(
                pl.col(self.counter_column)
                .list.eval(pl.element().struct.field("time"))
                .alias(self.name),
            )
        return data
