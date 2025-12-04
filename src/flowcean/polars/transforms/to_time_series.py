import polars as pl
from typing_extensions import override

from flowcean.core import Transform


class ToTimeSeries(Transform):
    time_feature: str | dict[str, str]

    def __init__(self, time_feature: str | dict[str, str]) -> None:
        super().__init__()
        self.time_feature = time_feature

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if isinstance(self.time_feature, str):
            time_feature = {
                self.time_feature: data.drop(self.time_feature)
                .collect_schema()
                .names(),
            }
        else:
            time_feature = self.time_feature

        return data.select(
            (
                pl.struct(
                    pl.col(t_feature).alias("time"),
                    pl.struct(pl.col(values)).alias("value"),
                ).implode()
                for t_feature, values in time_feature.items()
            ),
        )
