from collections.abc import Iterable, Iterator
from typing import override

import polars as pl

from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.environment.offline import OfflineEnvironment
from flowcean.core.environment.stepable import Finished


class ChainedOfflineEnvironments(IncrementalEnvironment):
    environments: Iterator[OfflineEnvironment]
    element: OfflineEnvironment

    def __init__(self, environments: Iterable[OfflineEnvironment]) -> None:
        self.environments = iter(environments)
        self.element = next(self.environments)

    @override
    def observe(self) -> pl.DataFrame:
        return self.element.observe()

    @override
    def step(self) -> None:
        try:
            self.element = next(self.environments)
        except StopIteration:
            raise Finished from StopIteration
