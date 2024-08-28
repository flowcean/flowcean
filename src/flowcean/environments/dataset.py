from __future__ import annotations

from collections.abc import Collection, Iterable
from itertools import islice
from typing import Any, override

import polars as pl
from tqdm import tqdm

from flowcean.core.environment.offline import OfflineEnvironment


class Dataset(OfflineEnvironment):
    data: pl.DataFrame

    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data

    @override
    def observe(self) -> pl.DataFrame:
        return self.data

    def __len__(self) -> int:
        return len(self.data)


def collect(
    environment: Iterable[pl.DataFrame] | Collection[pl.DataFrame],
    n: int | None = None,
    *,
    progress_bar: bool | dict[str, Any] = True,
) -> Dataset:
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
