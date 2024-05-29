import logging

import polars as pl
from numpy import interp

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class MatchSamplingRate(Transform):
    """Matches the sampling rate of all time series in the DataFrame.

    Interpolates the time series to match the sampling rate of the reference
    time series. The below example shows the usage of a `MatchSamplingRate`
    transform in an `experiment.yaml` file. Assuming the loaded data is
    represented by the table:

    | time_feature_a | feature_a | time_feature_b | feature_b | constant |
    | -------------- | --------- | -------------- | --------- | -------- |
    | [0, 1, 2]      | [2, 1, 7] | [0, 2]         | [10, 20]  | 1        |
    | [0, 1, 2]      | [4, 1, 0] | [0, 2]         | [20, 40]  | 2        |

    The following transform can be used to match the sampling rate
    of the time series described by `time_feature_b` and
    `feature_b` to the sampling rate of the time series described by
    `time_feature_a` and `feature_a`.

    ```
        transforms:
            - classpath: agenc.transforms.MatchSamplingRate
                arguments:
                    reference_timestamps: time_feature_a
                    feature_columns_with_timestamps:
                        feature_b: time_feature_b
            - ...
    ```

    The resulting Dataframe after the transform is:

    | time_feature_a | feature_a   | time_feature_b | feature_b    | constant |
    | -------------- | ----------- | -------------- | ------------ | -------- |
    | [0, 1, 2]      | [2, 1, 7]   | [0, 1, 2]       | [10, 15, 20]    | 1    |
    | [0, 1, 2]      | [4, 1, 0]   | [0, 1, 2]       | [20, 30, 40]    | 2    |



    Note that the used feature `time_feature_b` is still
    present in the DataFrame. To remove it use the `select` transform.
    """

    def __init__(
        self,
        reference_timestamps: str,
        feature_columns_with_timestamps: dict[str, str],
    ) -> None:
        """Initialize the MatchSamplingRate transform.

        Args:
            reference_timestamps: Timestamps of the reference feature.
            feature_columns_with_timestamps: Names of the features that are
                getting interpolated with their respective original timestamp
                feature names.
        """
        self.reference_timestamps = reference_timestamps
        self.feature_columns_with_timestamps = feature_columns_with_timestamps

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        logger.debug("Matching sampling rate of time series.")

        for i in range(len(data)):
            reference_timestamps = data[self.reference_timestamps][i]
            for (
                feature,
                timestamp,
            ) in self.feature_columns_with_timestamps.items():
                timestamps = data[timestamp][i].to_numpy()
                feature_data = data[feature][i].to_numpy()
                resampled_timeseries = interp(
                    reference_timestamps,
                    timestamps,
                    feature_data,
                ).tolist()
                data = (
                    data.lazy()
                    .with_row_index()
                    .with_columns(
                        pl.when(pl.col("index") == i)
                        .then(resampled_timeseries)
                        .otherwise(pl.col(feature))
                        .alias(feature)
                    )
                    .drop("index")
                    .collect()
                )
                reference_timestamps = pl.Series(
                    reference_timestamps
                ).to_list()
                data = (
                    data.lazy()
                    .with_row_index()
                    .with_columns(
                        pl.when(pl.col("index") == i)
                        .then(reference_timestamps)
                        .otherwise(pl.col(timestamp))
                        .alias(timestamp)
                    )
                    .drop("index")
                    .collect()
                )
        return data
