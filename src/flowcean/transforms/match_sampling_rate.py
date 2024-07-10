import logging

import polars as pl
import polars.selectors as cs

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class MatchSamplingRate(Transform):
    """Matches the sampling rate of all time series in the DataFrame.

    Interpolates the time series to match the sampling rate of the reference
    time series. The below example shows the usage of a `MatchSamplingRate`
    transform in a `run.py` file. Assuming the loaded data is
    represented by the table:

    | feature_a                         | feature_b                          | const |
    | ---                               | ---                                | ---   |
    | list[struct[datetime[us],struct[]]| list[struct[datetime[us],struct[]] | int   |
    | ----------------------------------| -----------------------------------| ----- |
    | [{2024-06-25 12:26:01.0,{1.2}},   | [{2024-06-25 12:26:00.0,{1.0}},    | 1     |
    |  {2024-06-25 12:26:02.0,{2.4}},   |  {2024-06-25 12:26:05.0,{2.0}}]    |       |
    |  {2024-06-25 12:26:03.0,{3.6}},   |                                    |       |
    |  {2024-06-25 12:26:04.0,{4.8}}]   |                                    |       |

    The following transform can be used to match the sampling rate
    of the time series `feature_b` to the sampling rate
    of the time series `feature_a`.

    ```
        ...
        environment.load()
        data = environment.get_data()
        transform = MatchSamplingRate(
            reference_feature="feature_a",
            feature_columns={
                "feature_b": "linear",
            },
        )
        transformed_data = transform.transform(data)
        ...
    ```

    The resulting Dataframe after the transform is:

    | feature_a                          | feature_b                          | const |
    | ---                                | ---                                | ---   |
    | list[struct[datetime[us],struct[]] | list[struct[datetime[us],struct[]] | int   |
    | -----------------------------------| -----------------------------------| ----- |
    | [{2024-06-25 12:26:01.0,{1.2}},    | [{2024-06-25 12:26:00.0,{1.0}},    | 1     |
    |  {2024-06-25 12:26:02.0,{2.4}},    |  {2024-06-25 12:26:01.0,{1.2}},    |       |
    |  {2024-06-25 12:26:03.0,{3.6}},    |  {2024-06-25 12:26:02.0,{2.4}},    |       |
    |  {2024-06-25 12:26:04.0,{4.8}}]    |  {2024-06-25 12:26:03.0,{3.6}}]    |       |


    """  # noqa: E501

    def __init__(
        self,
        reference_feature: str,
        feature_columns: dict[str, str],
    ) -> None:
        """Initialize the MatchSamplingRate transform.

        Args:
            reference_feature: Reference timeseries feature.
            feature_columns: key-value pairs of feature columns and
                the interpolation method to use. The interpolation
                method can be one of the following:
                - 'linear'
                - 'slerp' (to be implemented)
        """
        self.reference_feature = reference_feature
        self.feature_columns = feature_columns

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame.

        Args:
            data: Input DataFrame.

        Returns:
            Transformed DataFrame.

        """
        # preserve all constant columns that are not timeseries data
        non_timeseries_features = [
            col
            for col in data.columns
            if col not in [*self.feature_columns, self.reference_feature]
        ]
        logger.debug("Matching sampling rate of time series.")

        if self.reference_feature not in data.columns:
            msg = f"Reference {self.reference_feature} not found in the DataFrame."  # noqa: E501
            raise ValueError(msg)

        features = list(self.feature_columns.keys())
        reference_df = (
            data.select(pl.col(self.reference_feature).explode())
            .unnest(cs.all())
            .rename(
                lambda name: self.reference_feature + "_" + name
                if name != "time"
                else name
            )
        )
        result = pl.concat(
            [
                self.interpolate_feature(feature, data, reference_df)
                for feature in features
            ],
            how="horizontal",
        )

        return pl.concat(
            [
                data.select(self.reference_feature),
                result,
                data.select(non_timeseries_features),
            ],
            how="horizontal",
        )

    def interpolate_feature(
        self, feature: str, data: pl.DataFrame, reference_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Interpolate a single time series feature.

        Args:
            feature: Timeseries feature to interpolate.
            data: Input DataFrame.
            reference_df: Reference timeseries feature.

        Returns:
            Interpolated timeseries feature.

        """
        if self.feature_columns[feature] == "linear":
            feature_df = (
                data.select(pl.col(feature).explode())
                .unnest(cs.all())
                .unnest("value")
                .rename(
                    lambda name: feature + "_" + name
                    if name != "time"
                    else name
                )
            )
            result = (
                (
                    pl.concat([reference_df, feature_df], how="diagonal")
                    .sort("time")
                    .with_columns(
                        pl.col(feature_df.drop("time").columns).interpolate()
                    )
                    .drop_nulls()
                )
                .select(feature_df.columns)
                .select(
                    [
                        pl.struct(
                            pl.col("time"),
                            pl.struct(
                                {
                                    col: pl.col(col)
                                    for col in feature_df.columns
                                    if col != "time"
                                }
                            ).alias("value"),
                        ).alias(feature)
                    ]
                )
                .select(pl.all().implode())
            )
        return result
