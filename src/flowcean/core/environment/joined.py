from typing import Self, cast

import polars as pl

from .offline import OfflineEnvironment


class JoinedEnvironment(OfflineEnvironment):
    """Joins the features from multiple OfflineEnvironments.

    This environment joins the features of multiple OfflineEnvironments
    into a single one by concatenating them horizontally. All environments must
    have the same number of samples. If multiple environments share a feature,
    only the feature from the last environment will be used.
    """

    def __init__(self, *environments: OfflineEnvironment) -> None:
        """Initialize the extend environment.

        Args:
            *environments: List of offline environments whose features should
                be combined.
        """
        super().__init__()
        self.environments = environments

    def load(self) -> Self:
        for env in self.environments:
            env.load()
        return self

    def get_data(self) -> pl.DataFrame | pl.LazyFrame:
        """Get data from the environment.

        Returns:
            The loaded dataset.
        """
        data = [env.get_data() for env in self.environments]
        if any(isinstance(data_frame, pl.LazyFrame) for data_frame in data):
            return pl.concat(
                (data_frame.lazy() for data_frame in data),
                how="horizontal",
            )
        return pl.concat(cast(list[pl.DataFrame], data), how="horizontal")
