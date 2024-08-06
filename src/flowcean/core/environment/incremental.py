from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING, Any

import polars as pl
from tqdm import tqdm

if TYPE_CHECKING:
    from flowcean.environments.dataset import Dataset

from .base import Environment


class IncrementalEnvironment(Environment, Iterable[pl.DataFrame]):
    """Base class for incremental environments.

    An incremental environment loads data in a semi-interactive way, e.g.,
    stream, a sensor, etc. Data can be retrieved multiple times, but the
    environment cannot be controlled.
    """

    def collect(
        self,
        n: int,
        *,
        progress_bar: bool | dict[str, Any] = True,
    ) -> Dataset:
        """Collect n samples.

        Args:
            n: Number of samples to collect.
            progress_bar: Show a progress bar while collecting the samples. If
                a dictionary is passed, it is used as keyword arguments for the
                `tqdm` progress bar.

        Returns:
            Dataset with the collected data.
        """
        from flowcean.environments.dataset import Dataset

        samples = islice(self, n)

        if isinstance(progress_bar, dict):
            progress_bar.setdefault("desc", "Collecting samples")
            progress_bar.setdefault("total", n)
            samples = tqdm(
                samples,
                **progress_bar,
            )
        elif progress_bar:
            samples = tqdm(samples, desc="Collecting samples", total=n)

        data = pl.DataFrame()
        for sample in samples:
            data = data.vstack(sample)

        return Dataset(data)
