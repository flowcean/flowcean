from functools import reduce
from typing import Self

import polars as pl

from .offline import OfflineEnvironment


class StackEnvironment(OfflineEnvironment):
    """Combines multiple OfflineEnvironments into a single one.

    This environment combines multiple OfflineEnvironments into a single one by
    stacking them vertically. All environments need to share the identical
    features.
    """

    def __init__(self, *environments: OfflineEnvironment) -> None:
        super().__init__()
        self.environments = environments

    def load(self) -> Self:
        for env in self.environments:
            env.load()
        return self

    def get_data(self) -> pl.DataFrame:
        """Get data from the environment.

        Returns:
            The loaded dataset.
        """
        dfs = [env.get_data() for env in self.environments]
        return reduce(
            lambda x, y: x.vstack(y),
            dfs,
        )
