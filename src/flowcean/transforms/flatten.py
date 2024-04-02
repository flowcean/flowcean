import logging

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Flatten(Transform):
    """Flatten all timeseries in a DataFrame to individual features.

    The given DataFrame's timeseries are converted into individual features,
    with each time step creating a new feature. This transform will change the
    order of the columns in the resulting dataset.
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        for feature_name in data.columns:
            if data[feature_name].dtype != pl.Object:
                continue  # Skip any non timeseries features
            flatten_df = self.__flatten_feature__(data[feature_name])
            data = data.drop(feature_name).hstack(flatten_df)
        return data

    def __flatten_feature__(self, feature_series: pl.Series) -> pl.DataFrame:
        # First check if all entries have the same shape
        shapes = [sample.shape for sample in feature_series]

        if not all(shapes[0] == shape for shape in shapes[1:]):
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
                    f"{feature_series.name}_{index}": sample[index, 1]
                    for index in range(len(sample[:, 1]))
                }
                for sample in feature_series
            ],
        )
