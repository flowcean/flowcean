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
    time series. The `feature_interpolation_map` parameter is a dictionary that
    specifies the interpolation method for each feature. The keys are
    the feature names, and the values are the interpolation methods.
    The interpolation method can be 'linear' or 'nearest'. If the
    `feature_interpolation_map` parameter is not provided, all features
    except the reference feature will be interpolated using the
    'nearest' method. The `fill_strategy` parameter specifies the
    strategy to fill missing values after interpolation. The default
    value is 'both_ways', which means that missing values will be
    filled using both forward and backward filling. Other options
    include 'forward', 'backward', 'min', 'max', 'mean', 'zero', and
    'one'.The below example shows the usage of a `MatchSamplingRate`
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
        feature_interpolation_map: dict[str, MatchSamplingRateMethod]
        | None = None,
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
        if self.feature_interpolation_map is None:
            # use all columns that are not the reference feature with "nearest"
            # interpolation
            self.feature_interpolation_map = {
                feature: "nearest"
                for feature in data.columns
                if feature != self.reference_feature_name
            }
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
                    cast(
                        "Literal['linear', 'nearest']",
                        self.feature_interpolation_map[feature],
                    ),
                    cast("FillStrategy | None", self.fill_strategy),
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
    # Get the original time type from the reference feature
    original_time_type = reference_feature.schema["time"]

    # Cast time to Float64 for interpolation
    feature_df = feature_df.with_columns(pl.col("time").cast(pl.Float64))

    # Handle 'value' field
    if "value" in feature_df.columns:
        value_schema = feature_df.schema["value"]
        if isinstance(value_schema, pl.Struct):
            # Get the schema of 'value' and extract field names for structs
            original_field_names = [
                field.name for field in value_schema.fields
            ]
            feature_df = feature_df.unnest("value")
            value_is_struct = True
        else:
            # Rename non-struct 'value' to a temporary name to avoid conflicts
            feature_df = feature_df.rename(
                {"value": f"{target_feature_name}_value"},
            )
            original_field_names = [f"{target_feature_name}_value"]
            value_is_struct = False
    else:
        msg = f"Feature {target_feature_name} is missing 'value' field."
        raise ValueError(msg)

    # Store column names after unnesting (or renaming for non-struct 'value')
    value_columns = [col for col in feature_df.columns if col != "time"]

    # Get reference times and feature times, cast to Float64 for interpolation
    reference_times = reference_feature.get_column("time").cast(pl.Float64)
    feature_times = feature_df.get_column("time").cast(pl.Float64)

    # Combine all unique times and sort
    all_times = (
        pl.concat([reference_times, feature_times])
        .unique()
        .sort()
        .to_frame("time")
        .with_columns(pl.col("time").cast(pl.Float64))
    )

    # Join with feature data
    joined_df = all_times.join(feature_df, on="time", how="left")

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
    interpolated = interpolated.filter(
        pl.col("time").is_in(reference_times.implode()),
    )

    # Restore original time type
    interpolated = interpolated.with_columns(
        pl.col("time").cast(original_time_type),
    )

    # Restructure to nested format
    if value_is_struct:
        # Struct case: map original field names to their respective columns
        restructure_value = pl.struct(
            (
                pl.col(col).alias(name)
                for name, col in zip(
                    original_field_names,
                    value_columns,
                    strict=False,
                )
            ),
        ).alias("value")
    else:
        # Scalar case: restore the original scalar 'value' field
        restructure_value = pl.col(value_columns[0]).alias("value")

    restructure = pl.struct(pl.col("time"), restructure_value).alias(
        target_feature_name,
    )

    return interpolated.select(restructure).select(pl.all().implode())


class FeatureNotFoundError(Exception):
    """Feature not found in the DataFrame.

    This exception is raised when a feature is not found in the DataFrame.
    """

    def __init__(self, feature: str) -> None:
        super().__init__(f"{feature} not found")
