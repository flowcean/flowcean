import logging

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class SliceTimeSeries(Transform):
    """Slices a time series at given slice points.

    The transform takes two columns as input: 'time_series' and 'slice_points.'
    The 'time_series' column is sliced at the points specified by the 'time'
    entry of the 'slice_points' column. The result is a new time series column
    where each entry contains only the values from the original time series
    that fall between the specified slice points.

    Suppose you have a dataframe with a single line entry as follows:

    |time_series                |slice_points               |
    |---------------------------|---------------------------|
    |[(00:00:03, 1),            |[(00:00:05, 0),            |
    | (00:00:04, 2),            | (00:00:08, 1)]            |
    | (00:00:06, 7),            |                           |
    | (00:00:09, 0)]            |                           |

    Applying the slice time series transform results in a multi-line dataframe,
    where each line corresponds to a slice point:

    |time_series                |slice_points               |
    |---------------------------|---------------------------|
    |[(00:00:03, 1),            |[(00:00:05, 0)]            |
    | (00:00:04, 2)]            |                           |
    |[(00:00:06, 7)]            |[(00:00:08, 1)]            |

    The transform operates line-wise, meaning that each line in the input
    dataframe is processed independently. The resulting dataframe will have
    multiple lines depending on the number of slice points specified in each
    line.

    """

    def __init__(self, time_series: str, slice_points: str) -> None:
        """Initialize the SliceTimeSeries transform.

        Args:
            time_series: the time series column to slice.
            slice_points: the column that specifies the slice points.

        """
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
