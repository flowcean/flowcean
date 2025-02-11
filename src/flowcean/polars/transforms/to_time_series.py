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
                feature_name: self.time_feature
                for feature_name in data.collect_schema().names()
                if feature_name != self.time_feature
            }
        else:
            time_feature = self.time_feature

        return data.select(
            [
                pl.struct(
                    pl.col(t_feature).alias("time"),
                    pl.col(value_feature).alias("value"),
                )
                .implode()
                .alias(value_feature)
                for value_feature, t_feature in time_feature.items()
            ],
        )
