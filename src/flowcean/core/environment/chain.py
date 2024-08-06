from typing import Self, override

import polars as pl

from .offline import OfflineEnvironment


class ChainEnvironment(OfflineEnvironment):
    """Chains multiple OfflineEnvironments into a single one.

    When retrieving data, this environment concatenates the data vertically.
    This environment is useful when you want to chain multiple datasets into a
    single one. It is useful when you have multiple datasets that you want to
    use together.
    """

    def __init__(self, *environments: OfflineEnvironment) -> None:
        super().__init__()
        self.environments = environments

    @override
    def load(self) -> Self:
        for environment in self.environments:
            environment.load()
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        return pl.concat(
            [env.get_data() for env in self.environments], how="vertical"
        )
