import logging
import math
from typing import Any, cast, override

import numpy as np
import numpy.typing as npt
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

            data = data.with_columns(
                pl.DataFrame(
                    {
                        feature: [
                            resample_data(
                                time[row_index, :][0],
                                value[row_index, :][0],
                                dt,
                            )
                            for row_index in range(len(data))
                        ]
                    }
                )
            )
        return data


def resample_data(
    time: npt.NDArray[np.float64],
    value: npt.NDArray[np.float64],
    dt: float,
) -> Any:
    # Create the new time vector
    time_start = cast(float, time[0])
    time_end = cast(float, time[-1])
    t_interp = np.linspace(
        time_start, time_end, int(math.ceil(time_end - time_start) / dt) + 1
    )

    # Interpolate the value vector to match the new time vector
    value_interp = np.interp(t_interp, time, value)
    return (
        pl.DataFrame({"time": t_interp, "value": value_interp})
        .to_struct()
        .implode()
        .item()
    )


def is_timeseries_column(df: pl.DataFrame, column_name: str) -> bool:
    data_type = df.select(column_name).dtypes[0]

    if data_type.base_type() != pl.List:
        return False

    inner_type: pl.DataType = cast(pl.DataType, cast(pl.List, data_type).inner)
    if inner_type.base_type() != pl.Struct:
        return False

    field_names = [field.name for field in cast(pl.Struct, inner_type).fields]
    return "time" in field_names and "value" in field_names
