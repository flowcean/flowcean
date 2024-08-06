from __future__ import annotations

from abc import abstractmethod

import polars as pl

from .base import Environment
from .streaming import StreamingOfflineData


class OfflineEnvironment(Environment):
    """Base class for offline environments.

    An offline environment loads data in an non-interactive way, e.g., from a
    file, a database, etc. Data can only be retrieved once.
    """

    @abstractmethod
    def get_data(self) -> pl.DataFrame:
        """Get data from the environment.

        Returns:
            The loaded dataset.
        """

    def as_stream(self, batch_size: int = 1) -> StreamingOfflineData:
        """Get a streaming interface to the data of the environment.

        Args:
            batch_size: The number of samples to yield at each iteration.

        Returns:
            A streaming offline data.
        """
        return StreamingOfflineData(self, batch_size)

    def chain(self, other: OfflineEnvironment) -> OfflineEnvironment:
        """Combine this environment with another one vertically.

        Args:
            other: The environment to append vertically.

        Returns:
            The combined environment.
        """
        # prevent circular imports
        from .chain import ChainEnvironment

        return ChainEnvironment(self, other)

    def join(self, other: OfflineEnvironment) -> OfflineEnvironment:
        """Joins this environment with another one horizontally.

        Args:
            other: The environment to join horizontally.

        Returns:
            The joined environment.
        """
        # prevent circular imports
        from .joined import JoinedEnvironment

        return JoinedEnvironment(self, other)

    def to_time_series(
        self, time_feature: str | dict[str, str]
    ) -> OfflineEnvironment:
        """Convert this environment to a time series.

        Args:
            time_feature: The feature in this environment that represents the
                time vector. Either a string if all series share a common time
                vector, or a dictionary where the keys are the value features
                and the values are the corresponding time vector feature names.

        Returns:
            A OfflineEnvironment with exactly one sample containing the source
            environment as a time series.
        """
        from flowcean.environments.dataset import Dataset

        # Get the underlying dataframe
        data = self.get_data()
        # Create the time feature mapping
        if isinstance(time_feature, str):
            time_feature = {
                feature_name: time_feature
                for feature_name in data.columns
                if feature_name != time_feature
            }

        # Convert the features into a time series
        return Dataset(
            data.select(
                [
                    pl.struct(
                        pl.col(t_feature).alias("time"),
                        pl.col(value_feature).alias("value"),
                    )
                    .implode()
                    .alias(value_feature)
                    for value_feature, t_feature in time_feature.items()
                ]
            )
        )
