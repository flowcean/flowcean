import logging
from collections.abc import Iterable
from typing import cast, override

import polars as pl

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Flatten(Transform):
    """Flatten all time series in a DataFrame to individual features.

    The given DataFrame's time series are converted into individual features,
    with each time step creating a new feature. This transform will change the
    order of the columns in the resulting dataset.

    For example the dataset

    series_data              | A  | B
    -------------------------|----|----
    {[0, 0], [1, 1], [2, 2]} | 42 | 43
    {[0, 3], [1, 4], [2, 5]} | 44 | 45

    gets flattend into the dataset

    series_data_0 | series_data_1 | series_data_2 | A  | B
    --------------|---------------|---------------|----|----
    0             | 1             | 2             | 42 | 43
    3             | 4             | 5             | 42 | 43
    """

    def __init__(self, features: Iterable[str] | None = None) -> None:
        self.features = features

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        # Loop over the features we want to explode
        feature_names = (
            self.features
            if self.features is not None
            else [
                column_name
                for column_name in data.columns
                if is_timeseries_column(data, column_name)
            ]
        )

        for feature in feature_names:
            # Check if the feature really is a time series
            if not is_timeseries_column(data, feature):
                msg = f"Feature '{feature}' is no time series"
                raise NoTimeSeriesFeatureError(msg)

            # Figure out how "long" the feature is and how many new columns
            # need to be added
            row_lengths = data.select(
                pl.col(feature)
                .list.eval(pl.first().struct.field("value"))
                .list.len()
            ).unique()

            # Check if all rows have the same length
            if row_lengths.count().item(0, 0) > 1:
                msg = f"Time series length in feature '{feature}' varies"
                raise FeatureLengthVaryError(msg)
            n = row_lengths.item(0, 0)

            # Construct the new columns and drop the old feature
            data = data.with_columns(
                [
                    pl.col(feature)
                    .list.eval(pl.first().struct.field("value"))
                    .list.get(i)
                    .alias(f"{feature}_{i}")
                    for i in range(n)
                ]
            ).drop(feature)

        return data


class FeatureLengthVaryError(Exception):
    """Length of a feature varies over different rows."""


class NoTimeSeriesFeatureError(Exception):
    """Feature is no time series."""


def is_timeseries_column(df: pl.DataFrame, column_name: str) -> bool:
    data_type = df.select(column_name).dtypes[0]

    if data_type.base_type() != pl.List:
        return False

    inner_type: pl.DataType = cast(pl.DataType, cast(pl.List, data_type).inner)
    if inner_type.base_type() != pl.Struct:
        return False

    field_names = [field.name for field in cast(pl.Struct, inner_type).fields]
    return "time" in field_names and "value" in field_names
