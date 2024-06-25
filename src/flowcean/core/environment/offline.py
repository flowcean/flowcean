from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from .base import Environment
from .streaming import StreamingOfflineData

if TYPE_CHECKING:
    import polars as pl


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

    def __and__(self, other: OfflineEnvironment) -> OfflineEnvironment:
        """Combine this environment with another one vertically.

        Args:
            other: The environment to append vertically.

        Returns:
            The combined environment.
        """
        return self.stack(other)
