import logging
import math
from typing import Literal, TypeAlias, cast

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline
from typing_extensions import override

from flowcean.core import Transform
from flowcean.polars.is_time_series import is_timeseries_feature

logger = logging.getLogger(__name__)

InterpolationMethod: TypeAlias = Literal["linear", "cubic"]


class Resample(Transform):
    """Resample time series features to a given sampling rate."""

    def __init__(
        self,
        sampling_rate: float | dict[str, float],
        *,
        interpolation_method: InterpolationMethod = "linear",
    ) -> None:
        """Initializes the Resample transform.

        Args:
            sampling_rate: Target sampling rate for time series features. If a
                float is provided, all possible time series features will be
                resampled. Alternatively, a dictionary can be provided where
                the key is the feature and the value is the target sample rate.
            interpolation_method: The interpolation method to use. Supported
                are "linear" and "cubic", with the default being
                "linear".
        """
        self.sampling_rate = sampling_rate
        self.interpolation_method = interpolation_method

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        sampling_mapping = (
            {
                column_name: self.sampling_rate
                for column_name in data.collect_schema().names()
                if is_timeseries_feature(data, column_name)
            }
            if isinstance(self.sampling_rate, float)
            else cast(dict[str, float], self.sampling_rate)
        )

        for feature, dt in sampling_mapping.items():
            data = data.with_columns(
                pl.struct(
                    pl.col(feature)
                    .list.eval(pl.first().struct.field("time"))
                    .alias("time"),
                    pl.col(feature)
                    .list.eval(pl.first().struct.field("value"))
                    .alias("value"),
                )
                .map_elements(
                    lambda series, dt=dt: self.resample_data(
                        series,
                        dt,
                    ),
                    return_dtype=pl.List(
                        pl.Struct(
                            {
                                "time": pl.Float64,
                                "value": pl.Float64,
                            },
                        ),
                    ),
                )
                .alias(feature),
            )
        return data

    def resample_data(
        self,
        data: dict[str, list[float]],
        dt: float,
    ) -> pl.Series:
        time = data["time"]
        value = data["value"]

        time_start = time[0]
        time_end = time[-1]
        t_interp = np.linspace(
            time_start,
            time_end,
            int(math.ceil(time_end - time_start) / dt) + 1,
        )

        # Interpolate the value vector to match the new time vector
        value_interp = None
        if self.interpolation_method == "linear":
            value_interp = np.interp(t_interp, time, value)
        elif self.interpolation_method == "cubic":
            interpolator = CubicSpline(time, value)
            value_interp = interpolator(t_interp)
        else:
            logger.warning(
                "Unknown interpolation method %s. Defaulting to linear",
                self.interpolation_method,
            )
        return (
            pl.DataFrame({"time": t_interp, "value": value_interp})
            .to_struct()
            .implode()
            .item()
        )
