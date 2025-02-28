import logging
from collections.abc import Iterable
from typing import Literal, TypeAlias

import polars as pl
from scipy.signal import butter, sosfilt
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)

SignalFilterType: TypeAlias = Literal["lowpass", "highpass"]


class SignalFilter(Transform):
    """Applies a Butterworth filter to time series features.

    Applies a Butterworth lowpass or highpass filter to time series
    features. For this transform to work, the time series must already have a
    uniform sampling rate. Use a `Resample' transform to uniformly sample the
    points of a time series.
    """

    def __init__(
        self,
        features: Iterable[str],
        filter_type: SignalFilterType,
        filter_frequency: float,
        *,
        order: int = 5,
    ) -> None:
        """Initializes the Filter transform.

        Args:
            features: Features that shall be filtered.
            filter_type: Type of the filter to apply. Valid options are
                "lowpass" and "highpass".
            filter_frequency: Characteristic frequency of the filter in Hz. For
                high- and lowpass this is the cutoff frequency.
            order: Order of the Butterworth filter to uses. Defaults to 5.
        """
        self.features = features
        self.filter_type = filter_type
        self.frequency = filter_frequency
        self.order = order

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        for feature in self.features:
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
                    lambda series: self.filter_data(
                        series,
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

    def filter_data(self, data: dict[str, list[float]]) -> pl.Series:
        time = data["time"]
        value = data["value"]

        dt = time[1] - time[0]

        if self.filter_type == "lowpass":
            filter_coeffs = butter(
                self.order,
                self.frequency,
                "lowpass",
                fs=1 / dt,
                output="sos",
            )
        elif self.filter_type == "highpass":
            filter_coeffs = butter(
                self.order,
                self.frequency,
                "highpass",
                fs=1 / dt,
                output="sos",
            )
        else:
            logger.warning(
                "Unknown filter method %s",
                self.filter_type,
            )
            return (
                pl.DataFrame({"time": time, "value": value})
                .to_struct()
                .implode()
                .item()
            )

        # Apply the filter to the data
        value_filtered = sosfilt(filter_coeffs, value)

        return (
            pl.DataFrame({"time": time, "value": value_filtered})
            .to_struct()
            .implode()
            .item()
        )
