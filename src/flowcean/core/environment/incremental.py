from __future__ import annotations

import itertools
from abc import abstractmethod
from collections.abc import Iterable
from functools import reduce
from typing import TYPE_CHECKING, cast

import polars as pl

from .base import Environment

if TYPE_CHECKING:
    from collections.abc import Generator


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

    def take(self, n: int) -> pl.DataFrame:
        """Takes n batches from the environment.

        This method takes n batches of data from the environment and returns
        them as a single dataframe.

        Args:
            n: Number of data batches to be taken.

        Returns:
            Combined data frame from all individual batches.
        """
        return reduce(
            lambda x, y: x.vstack(y),
            cast(
                Iterable[pl.DataFrame],
                itertools.islice(self.get_next_data(), n),
            ),
        ).rechunk()
