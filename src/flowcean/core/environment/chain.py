from typing import Self, cast, override

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
    def get_data(self) -> pl.DataFrame | pl.LazyFrame:
        data = [env.get_data() for env in self.environments]
        if any(isinstance(data_frame, pl.LazyFrame) for data_frame in data):
            return pl.concat(
                (data_frame.lazy() for data_frame in data),
                how="vertical",
            )
        return pl.concat(cast(list[pl.DataFrame], data), how="vertical")
