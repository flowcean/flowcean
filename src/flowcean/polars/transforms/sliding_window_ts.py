import logging
from collections.abc import Iterable

import polars as pl
from typing_extensions import override

from flowcean.core import Transform
from flowcean.polars.is_time_series import is_timeseries_feature

logger = logging.getLogger(__name__)


class TimeSeriesSlidingWindow(Transform):
    """Convert single large time series into a set of smaller sub-series.

    Applies a sliding window to each individual time series sample of all or
    selected time series features while leaving other features unchanged.
    As a result, the resulting data frame will contain multiple samples for
    each original sample, where each sample is a sub-series of the original
    time series. The number of features (columns) will remain the same.
    For this transform to work, all selected time series features of a sample
    must have the same time vector. Use a `MatchSamplingRate` or `Resample`
    transform to ensure this is the case.
    """

    def __init__(
        self,
        window_size: int,
        *,
        features: str | Iterable[str] | None = None,
        stride: int = 1,
        rechunk: bool = True,
    ) -> None:
        """Initializes the TimeSeriesSlidingWindow transform.

        Args:
            window_size: The size of the sliding window.
            features: The features to apply the sliding window to. If None, all
                time series features are selected.
            stride: The stride of the sliding window.
            rechunk: Whether to rechunk the data after applying the transform.
                Rechunking improves performance of subsequent operations, but
                increases memory usage and may slow down the initial operation.
        """
        self.window_size = window_size
        self.features = features
        self.stride = stride
        self.rechunk = rechunk

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        # Find the target features the rolling window should be applied to.
        schema = data.collect_schema()
        if self.features is None:
            target_features = schema.names()
        elif isinstance(self.features, str):
            target_features = [self.features]
        else:
            target_features = list(self.features)

        # We can only handle time series data.
        target_features = [
            feature
            for feature in target_features
            if is_timeseries_feature(schema, feature)
        ]

        return data.map_batches(
            lambda df: self._map_frame(df, target_features=target_features),
            slice_pushdown=False,
            streamable=True,
        )

    def _map_frame(
        self,
        df: pl.DataFrame,
        target_features: list[str],
    ) -> pl.DataFrame:
        result_df = pl.DataFrame(schema=df.schema)
        all_features = df.schema.names()

        for row in df.iter_rows(named=True):
            # Find the length of the time series in the target features and
            # check that they match.
            ts_lengths = [len(row[feature]) for feature in target_features]
            if len(set(ts_lengths)) != 1:
                msg = "All time series features must have the same length."
                raise ValueError(msg)

            # Slice the time series features into windows, create a new row and
            # append it to the result.
            result_df = pl.concat(
                [
                    result_df,
                    *[
                        pl.DataFrame(
                            [
                                {
                                    feature: (
                                        row[feature][i : i + self.window_size]
                                        if feature in target_features
                                        else row[feature]
                                    )
                                    for feature in all_features
                                },
                            ],
                            schema=df.schema,
                        )
                        for i in range(
                            0,
                            ts_lengths[0] - self.window_size + 1,
                            self.stride,
                        )
                    ],
                ],
            )
        return result_df.rechunk() if self.rechunk else result_df
