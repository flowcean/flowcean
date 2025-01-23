from typing import override

import polars as pl

from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.environment.offline import OfflineEnvironment
from flowcean.core.environment.stepable import Finished


class StreamingOfflineEnvironment(IncrementalEnvironment):
    """Streaming offline environment.

    This environment streams data from an offline environment in batches.
    """

    environment: OfflineEnvironment
    batch_size: int
    data: pl.LazyFrame | None = None
    slice: pl.LazyFrame | None = None
    i: int = 0

    def __init__(
        self,
        environment: OfflineEnvironment,
        batch_size: int,
    ) -> None:
        """Initialize the streaming offline environment.

        Args:
            environment: The offline environment to stream.
            batch_size: The batch size of the streaming environment.
        """
        super().__init__()
        self.environment = environment
        self.batch_size = batch_size

    @override
    def _observe(self) -> pl.LazyFrame:
        if self.data is None:
            self.data = self.environment.observe()
        if self.slice is None:
            self.slice = self.data.slice(self.i, self.batch_size)
        return self.slice

    @override
    def step(self) -> None:
        if self.data is None:
            self.data = self.environment.observe()
        self.i += self.batch_size
        self.slice = self.data.slice(self.i, self.batch_size)
        if (
            self.slice.slice(0, 1)
            .collect(streaming=False)
            .select(pl.len())
            .item(0, 0)
            == 0
        ):
            raise Finished
