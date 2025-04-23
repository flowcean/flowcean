import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform


class SliceTimeSeries(Transform):
    """Slices time series features based on a counter column and deadzone."""

    def __init__(self, time_series: str, counter_feature: str) -> None:
        super().__init__()
        self.time_series = time_series
        self.counter_feature = counter_feature

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        time_series = (
            data.select(self.time_series)
            .with_row_index("experiment_i")
            .explode(self.time_series)
            .unnest(self.time_series)
        )
        slice_points = (
            data.select(self.counter_feature)
            .with_row_index("experiment_i")
            .explode(self.counter_feature)
            .unnest(self.counter_feature)
            .filter(pl.col("value").struct.field("data").diff() > 0)
            .drop("value")
            .with_columns(
                pl.int_range(pl.len()).over("experiment_i").alias("slice_i"),
            )
        )
        joined = time_series.join_asof(
            slice_points,
            on="time",
            by="experiment_i",
            strategy="forward",
            check_sortedness=False,
        )
        collapsed = (
            joined.group_by(["experiment_i", "slice_i"], maintain_order=True)
            .agg(
                pl.struct(pl.col("time"), pl.col("value")).alias(
                    self.time_series,
                ),
            )
            .drop("slice_i")
            .group_by("experiment_i")
            .agg(pl.col(self.time_series))
            .drop("experiment_i")
        )
        return pl.concat(
            [data.drop(self.time_series), collapsed],
            how="horizontal",
        )
