import logging
from typing import Literal, TypeAlias

import polars as pl
import polars.selectors as cs

from flowcean.core import Transform

logger = logging.getLogger(__name__)


MatchSamplingRateMethod: TypeAlias = Literal["linear"]


class MatchSamplingRate(Transform):
    """Matches the sampling rate of all time series in the DataFrame.

    Interpolates the time series to match the sampling rate of the reference
    time series. The below example shows the usage of a `MatchSamplingRate`
    transform in a `run.py` file. Assuming the loaded data is
    represented by the table:

    | feature_a                   | feature_b                   | const |
    | ---                         | ---                         | ---   |
    | list[struct[time,struct[]]] | list[struct[time,struct[]]] | int   |
    | --------------------------- | --------------------------- | ----- |
    | [{12:26:01.0, {1.2}},       | [{12:26:00.0, {1.0}},       | 1     |
    |  {12:26:02.0, {2.4}},       |  {12:26:05.0, {2.0}}]       |       |
    |  {12:26:03.0, {3.6}},       |                             |       |
    |  {12:26:04.0, {4.8}}]       |                             |       |

    The following transform can be used to match the sampling rate
    of the time series `feature_b` to the sampling rate
    of the time series `feature_a`.

    ```
        ...
        environment.load()
        data = environment.get_data()
        transform = MatchSamplingRate(
            reference_feature_name="feature_a",
            feature_interpolation_map={
                "feature_b": "linear",
            },
        )
        transformed_data = transform.transform(data)
        ...
    ```

    The resulting Dataframe after the transform is:

    | feature_a                   | feature_b                   | const |
    | ---                         | ---                         | ---   |
    | list[struct[time,struct[]]] | list[struct[time,struct[]]] | int   |
    | --------------------------- | --------------------------- | ----- |
    | [{12:26:00.0, {1.2}},       | [{12:26:00.0, {1.2}},       | 1     |
    |  {12:26:01.0, {2.4}},       |  {12:26:01.0, {1.4}},       |       |
    |  {12:26:02.0, {3.6}},       |  {12:26:02.0, {1.6}},       |       |
    |  {12:26:03.0, {4.8}}]       |  {12:26:03.0, {1.8}}]       |       |

    """

    def __init__(
        self,
        reference_feature_name: str,
        feature_interpolation_map: dict[str, MatchSamplingRateMethod],
    ) -> None:
        """Initialize the transform.

        Args:
            reference_feature_name: Reference timeseries feature.
            feature_interpolation_map: Key-value pairs of the timeseries
                features that are targeted in interpolation columns and the
                interpolation method to use. At the moment, the interpolation
                method can only be 'linear'.
        """
        self.reference_feature_name = reference_feature_name
        self.feature_interpolation_map = feature_interpolation_map

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Transform the input DataFrame.

        Args:
            data: Input DataFrame.

        Returns:
            Transformed DataFrame.

        """
        # preserve all constant columns that are not timeseries data
        transformed_data = pl.DataFrame()
        collected_data = data.collect()
        for i in range(len(collected_data.rows())):
            transformed_data_slice = self._transform_row(
                collected_data.slice(i, 1),
            )
            transformed_data = transformed_data.vstack(transformed_data_slice)
        return transformed_data.lazy()

    def _transform_row(self, data: pl.DataFrame) -> pl.DataFrame:
        non_timeseries_features = data.select(
            pl.exclude(
                *self.feature_interpolation_map.keys(),
                self.reference_feature_name,
            ),
        )
        debug_msg = (
            "Interpolating timeseries features: "
            f"{self.feature_interpolation_map.keys()} using the timestamps of "
            f"the and reference feature {self.reference_feature_name}"
        )
        logger.debug(debug_msg)

        if self.reference_feature_name not in data.columns:
            raise FeatureNotFoundError(feature=self.reference_feature_name)

        features = self.feature_interpolation_map.keys()

        reference_feature = (
            data.select(pl.col(self.reference_feature_name).explode())
            .unnest(cs.all())
            .rename({"value": self.reference_feature_name + "_value"})
        )

        result = pl.concat(
            [
                interpolate_feature(
                    feature,
                    data,
                    reference_feature,
                    self.feature_interpolation_map[feature],
                )
                for feature in features
            ],
            how="horizontal",
        )

        return pl.concat(
            [
                data.select(self.reference_feature_name),
                result,
                non_timeseries_features,
            ],
            how="horizontal",
        )


def interpolate_feature(
    target_feature_name: str,
    data: pl.DataFrame,
    reference_feature: pl.DataFrame,
    interpolation_method: str,
) -> pl.DataFrame:
    """Interpolate a single time series feature.

    Args:
        target_feature_name: Timeseries feature to interpolate.
        data: Input DataFrame.
        reference_feature: Reference timeseries feature.
        interpolation_method: Interpolation method to use.

    Returns:
        Interpolated timeseries feature.

    """
    logger.debug("Interpolating feature %s", target_feature_name)
    if interpolation_method == "linear":
        feature_df = (
            data.select(pl.col(target_feature_name).explode())
            .unnest(cs.all())
            .unnest("value")
            .rename(
                lambda name: target_feature_name + "_" + name
                if name != "time"
                else name,
            )
        )

        interpolated_features = (
            pl.concat(
                [reference_feature, feature_df],
                how="diagonal",
            )
            .sort("time")
            .with_columns(
                pl.col(feature_df.drop("time").columns).interpolate(),
            )
            .drop_nulls()
            .select(feature_df.columns)
        )
        restructure_to_time_series = pl.struct(
            pl.col("time"),
            pl.struct(
                {
                    col: pl.col(col)
                    for col in feature_df.columns
                    if col != "time"
                },
            ).alias("value"),
        )
        return interpolated_features.select(
            restructure_to_time_series.alias(target_feature_name),
        ).select(pl.all().implode())
    raise UnknownInterpolationError(interpolation_method=interpolation_method)


class FeatureNotFoundError(Exception):
    """Feature not found in the DataFrame.

    This exception is raised when a feature is not found in the DataFrame.
    """

    def __init__(self, feature: str) -> None:
        super().__init__(f"{feature} not found")


class UnknownInterpolationError(Exception):
    """Interpolation method is not implemented yet.

    This exception is raised when a feature is not found in the DataFrame.
    """

    def __init__(self, interpolation_method: str) -> None:
        super().__init__(f"{interpolation_method} not found")
