from __future__ import annotations

from collections.abc import Collection, Iterable
from itertools import islice
from typing import Any, override

import polars as pl
from tqdm import tqdm

from flowcean.core.environment.offline import OfflineEnvironment


class Dataset(OfflineEnvironment):
    """A dataset environment.

    This environment represents static tabular datasets.

    Attributes:
        data: The data to represent.
    """

    data: pl.LazyFrame
    _length: int | None = None

    def __init__(self, data: pl.DataFrame | pl.LazyFrame) -> None:
        """Initialize the dataset environment.

        Args:
            data: The data to represent.
        """
        if isinstance(data, pl.DataFrame):
            self.data = data.lazy()
            self._length = len(data)
        else:
            self.data = data
        super().__init__()

    @override
    def _observe(self) -> pl.LazyFrame:
        return self.data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._length is None:
            # This operation is potentially very slow / costly
            self._length = self.data.select(pl.len()).collect().item()
        return self._length


def collect(
    environment: Iterable[pl.LazyFrame] | Collection[pl.LazyFrame],
    n: int | None = None,
    *,
    progress_bar: bool | dict[str, Any] = True,
) -> Dataset:
    """Collect data from an environment.

    Args:
        environment: The environment to collect data from.
        n: Number of samples to collect. If None, all samples are collected.
        progress_bar: Whether to show a progress bar. If a dictionary is
            provided, it will be passed to the progress bar.

    Returns:
        The collected dataset.
    """
    from flowcean.environments.dataset import Dataset

    samples = islice(environment, n)

    if n is not None:
        total = n
    elif isinstance(environment, Collection):
        total = len(environment)
    else:
        total = None

    if isinstance(progress_bar, dict):
        progress_bar.setdefault("desc", "Collecting samples")
        progress_bar.setdefault("total", total)
        samples = tqdm(
            samples,
            **progress_bar,
        )
    elif progress_bar:
        samples = tqdm(samples, desc="Collecting samples", total=total)

    data = pl.concat(samples, how="vertical")
    return Dataset(data)
