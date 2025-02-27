import logging
from typing import Literal, TypeAlias, cast

import polars as pl
import polars.selectors as cs

from flowcean.core import Transform

logger = logging.getLogger(__name__)


MatchSamplingRateMethod: TypeAlias = Literal["linear", "nearest"]
FillStrategy: TypeAlias = Literal[
    "forward",
    "backward",
    "min",
    "max",
    "mean",
    "zero",
    "one",
    "both_ways",
]


class MatchSamplingRate(Transform):
    """Matches the sampling rate of all time series in the DataFrame.

    Interpolates the time series to match the sampling rate of the reference
    time series. The below example shows the usage of a `MatchSamplingRate`
    transform in a `run.py` file. Assuming the loaded data is
    represented by the table:

    ```
    | feature_a                   | feature_b                   | const |
    | ---                         | ---                         | ---   |
    | list[struct[time,struct[]]] | list[struct[time,struct[]]] | int   |
    | --------------------------- | --------------------------- | ----- |
    | [{12:26:01.0, {1.2}},       | [{12:26:00.0, {1.0}},       | 1     |
    |  {12:26:02.0, {2.4}},       |  {12:26:05.0, {2.0}}]       |       |
    |  {12:26:03.0, {3.6}},       |                             |       |
    |  {12:26:04.0, {4.8}}]       |                             |       |
    ```

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

    ```
    | feature_a                   | feature_b                   | const |
    | ---                         | ---                         | ---   |
    | list[struct[time,struct[]]] | list[struct[time,struct[]]] | int   |
    | --------------------------- | --------------------------- | ----- |
    | [{12:26:00.0, {1.2}},       | [{12:26:00.0, {1.2}},       | 1     |
    |  {12:26:01.0, {2.4}},       |  {12:26:01.0, {1.4}},       |       |
    |  {12:26:02.0, {3.6}},       |  {12:26:02.0, {1.6}},       |       |
    |  {12:26:03.0, {4.8}}]       |  {12:26:03.0, {1.8}}]       |       |
    ```

    """

    def __init__(
        self,
        reference_feature_name: str,
        feature_interpolation_map: dict[str, MatchSamplingRateMethod],
        fill_strategy: FillStrategy = "both_ways",
    ) -> None:
        """Initialize the transform.

        Args:
            reference_feature_name: Reference timeseries feature.
            feature_interpolation_map: Key-value pairs of the timeseries
                features that are targeted in interpolation columns and the
                interpolation method to use. The interpolation
                method can be 'linear' or 'nearest'.
            fill_strategy: Strategy to fill missing values after interpolation.
        """
        self.reference_feature_name = reference_feature_name
        self.feature_interpolation_map = feature_interpolation_map
        self.fill_strategy = fill_strategy

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
                    cast(FillStrategy | None, self.fill_strategy),
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
    interpolation_method: Literal["linear", "nearest"] = "linear",
    fill_strategy: FillStrategy | None = None,
) -> pl.DataFrame:
    """Interpolate a single time series feature using Polars expressions."""
    logger.debug("Interpolating feature %s", target_feature_name)

    # Extract and unnest feature dataframe
    feature_df = data.select(pl.col(target_feature_name).explode()).unnest(
        cs.all(),
    )
    # Handle scalar 'value' by wrapping into a struct
    if "value" in feature_df.columns:
        if not isinstance(feature_df.schema["value"], pl.Struct):
            # Wrap scalar 'value' into a struct with a single field
            feature_df = feature_df.with_columns(
                pl.struct([pl.col("value").alias("value")]).alias("value"),
            )
        # Now unnest the 'value' struct
        feature_df = feature_df.unnest("value")
    else:
        msg = f"Feature {target_feature_name} is missing 'value' field."
        raise ValueError(
            msg,
        )

    # Rename columns except 'time' to include the feature name
    feature_df = feature_df.rename(
        lambda name: f"{target_feature_name}_{name}"
        if name != "time"
        else name,
    )

    # Get reference times and feature times
    reference_times = reference_feature.get_column("time")
    feature_times = feature_df.get_column("time")

    # Combine all unique times and sort
    all_times = (
        pl.concat([reference_times, feature_times])
        .unique()
        .sort()
        .to_frame("time")
    )

    # Join with feature data
    joined_df = all_times.join(feature_df, on="time", how="left")

    # Get value columns (excluding time)
    value_columns = [col for col in feature_df.columns if col != "time"]

    # Interpolate missing values
    interpolated = joined_df.with_columns(
        [
            pl.col(col).interpolate(method=interpolation_method)
            for col in value_columns
        ],
    )
    if fill_strategy == "both_ways":
        fill_strategy = "backward"
        interpolated = interpolated.with_columns(
            [
                pl.col(col).fill_null(strategy=fill_strategy)
                for col in value_columns
            ],
        )
        fill_strategy = "forward"
        interpolated = interpolated.with_columns(
            [
                pl.col(col).fill_null(strategy=fill_strategy)
                for col in value_columns
            ],
        )
    elif fill_strategy:
        interpolated = interpolated.with_columns(
            [
                pl.col(col).fill_null(strategy=fill_strategy)
                for col in value_columns
            ],
        )

    # Filter to only include reference times
    interpolated = interpolated.filter(pl.col("time").is_in(reference_times))
    # Determine if the original 'value' was a scalar
    is_scalar_value = (
        len(value_columns) == 1
        and value_columns[0] == f"{target_feature_name}_value"
    )

    # Restructure to nested format, preserving scalar 'value' if needed
    if is_scalar_value:
        restructure_value = pl.col(value_columns[0]).alias("value")
    else:
        restructure_value = pl.struct(value_columns).alias("value")

    restructure = pl.struct(
        pl.col("time"),
        restructure_value,
    ).alias(target_feature_name)

    return interpolated.select(restructure).select(pl.all().implode())


class FeatureNotFoundError(Exception):
    """Feature not found in the DataFrame.

    This exception is raised when a feature is not found in the DataFrame.
    """

    def __init__(self, feature: str) -> None:
        super().__init__(f"{feature} not found")
