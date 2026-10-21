import math
from typing import override

import polars as pl

from flowcean.core import Finished, IncrementalEnvironment, OfflineEnvironment


class StreamingOfflineEnvironment(IncrementalEnvironment[pl.LazyFrame]):
    """Streaming offline environment.

    This environment streams data from an offline environment in batches.
    """

    environment: OfflineEnvironment[pl.LazyFrame]
    batch_size: int
    data: pl.LazyFrame | None = None
    slice: pl.LazyFrame | None = None
    i: int = 0

    def __init__(
        self,
        environment: OfflineEnvironment[pl.LazyFrame],
        batch_size: int,
        *,
        size: int | None = None,
    ) -> None:
        """Initialize the streaming offline environment.

        Args:
            environment: The offline environment to stream.
            batch_size: The batch size of the streaming environment.
            size: The number of samples in the environment. If provided, the
                number of steps will be calculated based on this value.
        """
        if batch_size <= 0:
            message = "batch_size must be positive"
            raise ValueError(message)
        self.environment = environment
        self.batch_size = batch_size
        self.sample_count = size

    @override
    def _observe(self) -> pl.LazyFrame:
        if self.data is None:
            self.data = self.environment.observe()
        if self.slice is None:
            batch = self.data.slice(self.i, self.batch_size)
            if batch.head(1).collect().is_empty():
                raise Finished
            self.slice = batch
        return self.slice

    @override
    def step(self) -> None:
        if self.data is None:
            self.data = self.environment.observe()
        self.i += self.batch_size
        self.slice = None
        self._observe()

    @override
    def num_steps(self) -> int | None:
        if self.sample_count is not None:
            return math.ceil(self.sample_count / self.batch_size)
        return None
