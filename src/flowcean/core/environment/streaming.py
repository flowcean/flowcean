from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Self, override

from .base import NotLoadedError
from .incremental import IncrementalEnvironment

if TYPE_CHECKING:
    from collections.abc import Generator

    import polars as pl

    from .offline import OfflineEnvironment


class StreamingOfflineData(IncrementalEnvironment):
    """Streaming offline data.

    This class wraps an offline environment and provides a streaming interface
    to its data. The data is loaded once and then streamed in batches of a
    given size.

    Attributes:
        environment: The offline environment to wrap.
        batch_size: The number of samples to yield at each iteration.
        index: The current index of the data.
        data: The loaded data.
    """

    environment: OfflineEnvironment
    batch_size: int
    index: int
    data: pl.DataFrame | None

    def __init__(
        self,
        environment: OfflineEnvironment,
        batch_size: int = 1,
    ) -> None:
        """Initialize a streaming offline data.

        Args:
            environment: The offline environment to wrap.
            batch_size: The number of samples to yield at each iteration.
        """
        self.environment = environment
        self.batch_size = batch_size
        self.index = 0
        self.data = None

    @override
    def load(self) -> Self:
        self.environment.load()
        self.data = self.environment.get_data()
        return self

    @override
    def get_next_data(self) -> Generator[pl.DataFrame, None, None]:
        if self.data is None:
            raise NotLoadedError
        for i in range(0, len(self.data), self.batch_size):
            yield self.data.slice(i, self.batch_size)
