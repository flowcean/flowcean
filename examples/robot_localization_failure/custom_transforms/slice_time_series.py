import logging

import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class SliceTimeSeries(Transform):
    def __init__(self, time_series: str, slice_points: str) -> None:
        super().__init__()
        self.time_series = time_series
        self.slice_points = slice_points

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug(
            "Slicing time series '%s' at points '%s'",
            self.time_series,
            self.slice_points,
        )
        time_series = (
            data.select(self.time_series)
            .with_row_index("experiment_i")
            .explode(self.time_series)
            .unnest(self.time_series)
        )
        slice_points = (
            data.select(self.slice_points)
            .with_row_index("experiment_i")
            .explode(self.slice_points)
            .filter(pl.col(self.slice_points) > 0)
            .select(
                pl.col("experiment_i"),
                pl.col(self.slice_points).alias("time"),
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
        ).explode(self.time_series)
