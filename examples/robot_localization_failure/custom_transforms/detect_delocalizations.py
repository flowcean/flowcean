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
            "Detecting delocalizations in column '%s' with output name '%s'",
            self.counter_column,
            self.name,
        )
        compute_difference = pl.element().struct.with_fields(
            pl.field("value").struct.field("data").diff().alias("value"),
        )
        filter_events = (
            pl.element()
            .struct.field("time")
            .filter(pl.element().struct.field("value") > 0)
        )
        return data.with_columns(
            pl.col(self.counter_column)
            .list.eval(compute_difference)
            .list.eval(filter_events)
            .alias(self.name),
        )
