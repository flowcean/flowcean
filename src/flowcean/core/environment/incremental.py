from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from .base import Environment

if TYPE_CHECKING:
    from collections.abc import Generator

    import polars as pl


class IncrementalEnvironment(Environment):
    """Base class for incremental environments.

    An incremental environment loads data in a semi-interactive way, e.g.,
    stream, a sensor, etc. Data can be retrieved multiple times, but the
    environment cannot be controlled.
    """

    @abstractmethod
    def get_next_data(self) -> Generator[pl.DataFrame, None, None]:
        """Get the next data from the environment.

        This method is a generator that yields the next batch of data from the
        environment. The generator should yield data until the environment is
        exhausted.

        Yields:
            The next batch of data.
        """
