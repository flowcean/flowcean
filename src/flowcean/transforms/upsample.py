import logging
from datetime import timedelta

import polars as pl

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Upsample(Transform):
    """Upsample a time series.

    Upsample a time series to a higher sampling rate using the polars method
    "upsample". The method can upsample to a higher sampling rate and fill
    missing values using interpolation.

    Args:
        time_column (str): The name of the time column.
        sampling_rate (str | timedelta): The sampling rate to upsample to.
        offset (str | timedelta): The offset to use for the upsampling.
    """

    def __init__(
        self,
        time_column: str,
        sampling_rate: str | timedelta,
        offset: str | timedelta,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.time_column = time_column
        self.offset = offset

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        logger.debug("Upsampling time series")
        return (
            data.upsample(
                time_column=self.time_column,
                every=self.sampling_rate,
                offset=self.offset,
            )
            .interpolate()
            .fill_null(strategy="forward")
        )
