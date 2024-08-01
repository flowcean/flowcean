import logging
from typing import cast, override

import numpy as np
import polars as pl

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Resample(Transform):
    """Resample time series features to a given sampling rate."""

    def __init__(self, sampling_rate: float | dict[str, float]) -> None:
        """Initializes the Resample transform.

        Args:
            sampling_rate: Target sampling rate for time series features. If a
            float is provided, all possible time series features will be
            resampled. Alternatively, a dictionary can be provided where the
            key is the feature and the value is the target sample rate.
        """
        self.sampling_rate = sampling_rate

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        sampling_mapping = (
            {
                column_name: self.sampling_rate
                for column_name in data.columns
                if is_timeseries_column(data, column_name)
            }
            if isinstance(self.sampling_rate, float)
            else cast(dict[str, float], self.sampling_rate)
        )

        for feature, dt in sampling_mapping.items():
            # Select the time and the value vector
            time = data.select(
                pl.col(feature).list.eval(pl.first().struct.field("time"))
            ).to_numpy()

            value = data.select(
                pl.col(feature).list.eval(pl.first().struct.field("value"))
            ).to_numpy()

            # Create the new time vector
            t_interp = np.arange(time[0], time[-1], dt)

            # Interpolate the value vector to match the new time vector
            value_interp = np.interp(t_interp, time, value)

            # Build a new feature from the interpolate data.
            # Because the new column has also the name "feature", the old
            # feature will be overriden
            data = data.with_columns(
                pl.DataFrame({"time": t_interp, "value": value_interp})
                .to_struct()
                .implode()
                .alias(feature)
            )

        return data


def is_timeseries_column(df: pl.DataFrame, column_name: str) -> bool:
    data_type = df.select(column_name).dtypes[0]

    if data_type.base_type() != pl.List:
        return False

    inner_type: pl.DataType = cast(pl.DataType, cast(pl.List, data_type).inner)
    if inner_type.base_type() != pl.Struct:
        return False

    field_names = [field.name for field in cast(pl.Struct, inner_type).fields]
    return "time" in field_names and "value" in field_names
