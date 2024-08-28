from __future__ import annotations

from typing import TYPE_CHECKING, Self, override

from flowcean.core.environment.base import NotLoadedError

from .incremental import IncrementalEnvironment

if TYPE_CHECKING:
    import polars as pl

    from .offline import OfflineEnvironment


class StreamingOfflineData[Observation: pl.DataFrame](IncrementalEnvironment):
    """Streaming offline data.

    This class wraps an offline environment and provides a streaming interface
    to its data. The data is loaded once and then streamed in batches of a
    given size.
    """

    environment: OfflineEnvironment[Observation]
    batch_size: int
    data: Observation | None = None
    i: int = 0

    def __init__(
        self,
        environment: OfflineEnvironment[Observation],
        batch_size: int = 1,
    ) -> None:
        """Initialize a streaming offline data.

        Args:
            environment: The offline environment to wrap.
            batch_size: The number of samples to yield at each iteration.
        """
        self.environment = environment
        self.batch_size = batch_size

    @override
    def load(self) -> Self:
        self.environment.load()
        self.data = self.environment.observe()
        return self

    @override
    def observe(self) -> pl.DataFrame:
        if self.data is None:
            raise NotLoadedError
        return self.data[self.i : self.i + self.batch_size]

    def step(self) -> None:
        if self.data is None:
            raise NotLoadedError
        next_i = self.i + self.batch_size
        if next_i > len(self.data):
            raise StopIteration
        self.i = next_i
