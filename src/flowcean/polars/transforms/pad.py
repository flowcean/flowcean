import logging
from collections.abc import Iterable
from typing import Any, cast

import polars as pl
from typing_extensions import override

from flowcean.core import Transform
from flowcean.polars.is_time_series import is_timeseries_feature
from flowcean.polars.time_series_type import (
    get_time_series_time_type,
    get_time_series_value_type,
)

logger = logging.getLogger(__name__)


class Pad(Transform):
    """Pad time-series features to the specified length.

    Pad time-series features to the specified end-time by holding their last
    value for one more sample.
    This is useful for ensuring that all time-series features cover at least a
    time interval of the specified length. Time-series that are already longer
    than the specified will not be modified.
    The resulting features will **not** be equidistant in time. To achieve
    equidistant time-series, consider using the `Resample` transform after
    padding.
    """

    def __init__(
        self,
        length: float,
        *,
        features: None | str | Iterable[str] = None,
    ) -> None:
        """Initializes the Pad transform.

        Args:
            length: The length (time) to pad the features to. This is the
                minimum length that the features will have after applying this
                transform.
            features: The features to apply this transform to. Defaults to
                `None`, which will apply the transform to all time-series
                features.
        """
        self.length = length
        self.features = (
            cast(
                "list[str]",
                ([features] if isinstance(features, str) else list(features)),
            )
            if features is not None
            else None
        )

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        # Select the features to pad
        target_features = []
        schema = data.collect_schema()
        if self.features is None:
            target_features = [
                name
                for name, _ in schema.items()
                if is_timeseries_feature(schema, name)
            ]
        else:
            target_features = self.features

        for feature in target_features:
            value_type = get_time_series_value_type(
                cast("pl.DataType", schema.get(feature)),
            )
            time_type = get_time_series_time_type(
                cast("pl.DataType", schema.get(feature)),
            )

            data = data.with_columns(
                pl.struct(
                    pl.col(feature)
                    .list.eval(pl.element().struct.field("time"))
                    .alias("time"),
                    pl.col(feature)
                    .list.eval(pl.element().struct.field("value"))
                    .alias("value"),
                )
                .map_elements(
                    lambda x: self.__map_elements__(x),
                    return_dtype=pl.List(
                        pl.Struct(
                            {
                                "time": time_type,
                                "value": value_type,
                            },
                        ),
                    ),
                )
                .alias(feature),
            )
        return data

    def __map_elements__(
        self,
        element: dict[str, list[Any]],
    ) -> pl.Series:
        value = element["value"]
        time = element["time"]

        if time[-1] < self.length:
            value.append(value[-1])
            time.append(self.length)

        return (
            pl.DataFrame({"time": time, "value": value})
            .to_struct()
            .implode()
            .item()
        )
