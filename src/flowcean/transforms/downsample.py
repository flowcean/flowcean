import logging
from datetime import timedelta

import polars as pl

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Downsample(Transform):
    """Downsample a time series.

    Downsample a time series to a lower sampling rate using the polars method
    "group_by_dynamic". The method can downsample to the first, last or mean
    value of the group.

    Args:
        time_column (str): The name of the time column.

        sampling_rate (str | timedelta): The sampling rate to downsample to.

        value (str): The value to use for the downsampling. Can be "first",
            "last" or "mean".


    """

    def __init__(
        self,
        time_column: str,
        sampling_rate: str | timedelta,
        value: str = "first",
    ) -> None:
        self.sampling_rate = sampling_rate
        self.time_column = time_column
        self.value = value

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        logger.debug("Downsampling time series")

        if self.value == "first":
            data = data.group_by_dynamic(
                self.time_column, every=self.sampling_rate
            ).agg(
                pl.col(x).first()
                for x in data.columns
                if x != self.time_column
            )
        elif self.value == "last":
            data = data.group_by_dynamic(
                self.time_column, every=self.sampling_rate
            ).agg(
                pl.col(x).last() for x in data.columns if x != self.time_column
            )
        elif self.value == "mean":
            data = data.group_by_dynamic(
                self.time_column, every=self.sampling_rate
            ).agg(
                pl.col(x).mean() for x in data.columns if x != self.time_column
            )
        return data
