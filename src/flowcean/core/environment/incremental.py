from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from itertools import islice
from typing import TYPE_CHECKING, Any, override

import polars as pl
from tqdm import tqdm

from flowcean.core.environment.observable import Observable
from flowcean.core.environment.stepable import Finished, Stepable

if TYPE_CHECKING:
    from flowcean.core.transform import Transform
    from flowcean.environments.dataset import Dataset


class IncrementalEnvironment(
    Observable[pl.DataFrame],
    Stepable,
    Iterable[pl.DataFrame],
):
    @override
    def __iter__(self) -> Iterator[pl.DataFrame]:
        yield self.observe()
        while True:
            try:
                self.step()
            except Finished:
                break
            yield self.observe()

    def collect(self) -> Dataset:
        return collect(self)

    def num_steps(self) -> int | None:
        return None

    def with_transform(
        self,
        transform: Transform,
    ) -> TransformedIncrementalEnvironment:
        return TransformedIncrementalEnvironment(self, transform)

    def __rshift__(
        self,
        transform: Transform,
    ) -> TransformedIncrementalEnvironment:
        return self.with_transform(transform)


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


class TransformedIncrementalEnvironment(IncrementalEnvironment):
    environment: IncrementalEnvironment
    transform: Transform

    def __init__(
        self,
        environment: IncrementalEnvironment,
        transform: Transform,
    ) -> None:
        self.environment = environment
        self.transform = transform

    @override
    def observe(self) -> pl.DataFrame:
        data = self.environment.observe()
        return self.transform(data)

    @override
    def step(self) -> None:
        self.environment.step()

    @override
    def num_steps(self) -> int | None:
        return self.environment.num_steps()

    def with_transform(
        self,
        transform: Transform,
    ) -> TransformedIncrementalEnvironment:
        transform = self.transform.chain(transform)
        return TransformedIncrementalEnvironment(self.environment, transform)
