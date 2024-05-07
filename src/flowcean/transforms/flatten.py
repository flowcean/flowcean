import logging
from collections.abc import Iterable
from typing import override

import polars as pl

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Flatten(Transform):
    """Flatten all time series in a DataFrame to individual features.

    The given DataFrame's time series are converted into individual features,
    with each time step creating a new feature. This transform will change the
    order of the columns in the resulting dataset.

    For example the dataset

    series_data | A  | B
    ------------|----|----
    [0, 1, 2]   | 42 | 43
    [3, 4, 5]   | 44 | 45

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
        if self.features is None:
            target_features = [
                feature_name
                for feature_name in data.columns
                if data[feature_name].dtype == pl.List
            ]
        else:
            target_features = list(self.features)

        for feature_name in target_features:
            data = data.drop(feature_name).hstack(
                self.__flatten_feature__(data[feature_name]),
            )

        return data.rechunk()

    def __flatten_feature__(self, feature_series: pl.Series) -> pl.DataFrame:
        # First check if all entries have the same shape
        lengths = [len(sample) for sample in feature_series]

        if not all(lengths[0] == length for length in lengths[1:]):
            logger.error(
                "Timeseries in %s has inconsistent length.",
                feature_series.name,
            )
            msg = (
                f"Timeseries in {feature_series.name} has inconsistent length."
            )
            raise RuntimeError(msg)

        return pl.DataFrame(
            [
                {
                    f"{feature_series.name}_{index}": sample[index]
                    for index in range(len(sample))
                }
                for sample in feature_series
            ],
        )
