import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform


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
        return data.with_columns(
            pl.col(self.counter_column)
            .list.eval(
                pl.element().struct.with_fields(
                    pl.field("value")
                    .struct.field("data")
                    .diff()
                    .alias("value"),
                ),
            )
            .list.eval(
                pl.element()
                .struct.field("time")
                .filter(pl.element().struct.field("value") > 0),
            )
            .alias(self.name),
        )
