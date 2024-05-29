from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from flowcean.environments.dataset import Dataset

from .base import Environment


class IncrementalEnvironment(Environment, Iterable[pl.DataFrame]):
    """Base class for incremental environments.

    An incremental environment loads data in a semi-interactive way, e.g.,
    stream, a sensor, etc. Data can be retrieved multiple times, but the
    environment cannot be controlled.
    """

    def collect(self, n: int) -> Dataset:
        """Collect n samples.

        Args:
            n: Number of samples to collect.

        Returns:
            Dataset with the collected data.
        """
        from flowcean.environments.dataset import Dataset

        data = pl.DataFrame()
        for sample in islice(self, n):
            data = data.vstack(sample)

        return Dataset(data)
