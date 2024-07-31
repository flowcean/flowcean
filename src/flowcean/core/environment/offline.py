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

    def stack(self, other: OfflineEnvironment) -> OfflineEnvironment:
        """Combine this environment with another one vertically.

        Args:
            other: The environment to append vertically.

        Returns:
            The combined environment.
        """
        # Necessary to prevent circular imports
        from .stack import StackEnvironment

        return StackEnvironment(self, other)

    def to_time_series(
        self, time_feature: str | dict[str, str]
    ) -> OfflineEnvironment:
        """Convert this envrionment to a time series.

        Args:
            time_feature: The feature in this environment that represents the
                time vector. Either a string if all series share a common time
                vector, or a dictionary where the keys are the value features
                and the values are the corresponding time vector feature names.

        Returns:
            A OfflineEnvrionment with exactly one sample containg the source
            environment as a time series.
        """
        from flowcean.environments.dataset import Dataset

        # Get the underlaying dataframe
        data = self.get_data()
        result_data = pl.DataFrame()
        # Create the time feature mapping
        if isinstance(time_feature, str):
            time_feature = {
                feature_name: time_feature
                for feature_name in data.columns
                if feature_name != time_feature
            }

        # Convert each feature into a time series
        for value_feature, t_feature in time_feature.items():
            result_data = result_data.with_columns(
                data.select(
                    [
                        t_feature,
                        value_feature,
                    ]
                )
                .rename(
                    {
                        t_feature: "time",
                        value_feature: "value",
                    }
                )
                .to_struct()
                .implode()
                .alias(value_feature)
            )

        return Dataset(result_data)
