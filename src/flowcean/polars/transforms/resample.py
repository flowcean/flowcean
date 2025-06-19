import logging
import math
from datetime import timedelta
from typing import Literal, TypeAlias, cast

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline
from typing_extensions import override

from flowcean.core import Transform
from flowcean.polars.is_time_series import is_timeseries_feature

logger = logging.getLogger(__name__)

InterpolationMethod: TypeAlias = Literal["linear", "cubic"]

_value_feature = "_value"
_time_feature = "_time"
_time_range_feature = "_time_range"
_index_feature = "_index"

_min_time_feature = "_min_time"
_max_time_feature = "_max_time"


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
        if interpolation_method not in {"linear", "cubic"}:
            msg = (
                f"Unsupported interpolation method: {interpolation_method}. "
                "Supported methods are 'linear' and 'cubic'.",
            )
            raise ValueError(msg)

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        sampling_mapping = (
            {
                column_name: self.sampling_rate
                for column_name in data.collect_schema().names()
                if is_timeseries_feature(data, column_name)
            }
            if isinstance(self.sampling_rate, float)
            else cast("dict[str, float]", self.sampling_rate)
        )

        for feature, dt in sampling_mapping.items():
            if self.interpolation_method == "linear":
                # Use polars' expressions for linear interpolation
                # Convert all time series to a "normal" dataframe with an
                # additional index feature
                exploded_feature_df = (
                    data.select(
                        [
                            pl.col(feature)
                            .list.eval(pl.element().struct.field("value"))
                            .alias(_value_feature),
                            pl.col(feature)
                            .list.eval(pl.element().struct.field("time"))
                            .alias(_time_feature),
                        ],
                    )
                    .with_row_index(name=_index_feature)
                    .explode([_value_feature, _time_feature])
                )

                # Convert the time feature to a polars time type if
                # it is not already
                if not exploded_feature_df.collect_schema()[
                    _time_feature
                ].is_temporal():
                    exploded_feature_df = exploded_feature_df.with_columns(
                        (pl.col(_time_feature) * 1_000_000_000).cast(
                            pl.Time(),
                        ),
                    )

                # Build the target time vectors for the feature.
                # Those are the times at which we want to
                # have the values after resampling.
                target_times_df = (
                    exploded_feature_df.select(
                        [
                            pl.col(_index_feature).over(_index_feature),
                            pl.col(_time_feature)
                            .min()
                            .alias(_min_time_feature)
                            .over(_index_feature),
                            pl.col(_time_feature)
                            .max()
                            .alias(_max_time_feature)
                            .over(_index_feature),
                        ],
                    )
                    .unique(subset=[_index_feature])
                    .select(
                        [
                            pl.col(_index_feature),
                            pl.time_ranges(
                                pl.col(_min_time_feature),
                                pl.col(_max_time_feature),
                                interval=timedelta(seconds=dt),
                            ).alias(_time_range_feature),
                        ],
                    )
                )

                # Extend the target times by the times from the
                # working DataFrame. This is necessary to include
                # the original data points in the interpolation.
                target_times_extended_df = (
                    pl.concat(
                        [
                            target_times_df.explode(
                                _time_range_feature,
                            ).rename(
                                {_time_range_feature: _time_feature},
                            ),
                            exploded_feature_df.select(
                                [
                                    pl.col(_index_feature),
                                    pl.col(_time_feature),
                                ],
                            ),
                        ],
                    )
                    # Remove duplicate entries. Those can occur if a original
                    # datapoint and a sampling timestep are the same.
                    .unique()
                    # Unfortunately, we need to sort the DataFrame
                    # to ensure that the interpolation works correctly.
                    .sort(
                        _index_feature,
                        _time_feature,
                    )
                )

                # Combine the target times with the working DataFrame
                # and interpolate the values
                joined_df = (
                    target_times_extended_df.join(
                        exploded_feature_df,
                        how="left",
                        on=[pl.col(_index_feature), pl.col(_time_feature)],
                    )
                    .with_columns(
                        pl.col(
                            _value_feature,
                        ).interpolate_by(
                            # Unpack the time feature to seconds
                            pl.col(_time_feature).dt.hour() * pl.lit(60 * 60)
                            + pl.col(_time_feature).dt.minute() * pl.lit(60)
                            + pl.col(_time_feature).dt.second(
                                fractional=True,
                            ),
                        ),
                    )
                    .fill_null(strategy="forward")
                )

                # Only keep the rows matching the target times - other rows
                # were just needed for the interpolation and are not part of
                # the resampled data
                joined_df = joined_df.join(
                    target_times_df,
                    on=pl.col(_index_feature),
                    how="left",
                ).filter(
                    pl.col(_time_feature).is_in(pl.col(_time_range_feature)),
                )

                # Convert the resampled data back to the time series format and
                # join it back to the rest of the data
                data = pl.concat(
                    [
                        data.drop(
                            feature,
                        ),  # `feature` comes from the resampled frame
                        joined_df.group_by(pl.col(_index_feature))
                        .agg(
                            pl.struct(
                                (
                                    # Convert the polars.Time back to a float
                                    # of seconds
                                    pl.col(_time_feature).dt.hour()
                                    * pl.lit(60 * 60)
                                    + pl.col(_time_feature).dt.minute()
                                    * pl.lit(60)
                                    + pl.col(_time_feature).dt.second(
                                        fractional=True,
                                    )
                                ).alias("time"),
                                pl.col(_value_feature).alias("value"),
                            )
                            .implode()
                            .alias(feature),
                        )
                        .drop(_index_feature),
                    ],
                    how="horizontal",
                )
            else:
                # Use map_elements with scipy for cubic interpolation
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
                        lambda series, dt=dt: self.cubic_resample(
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

    def cubic_resample(
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
        interpolator = CubicSpline(time, value)
        value_interp = interpolator(t_interp)
        return (
            pl.DataFrame({"time": t_interp, "value": value_interp})
            .to_struct()
            .implode()
            .item()
        )
