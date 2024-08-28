from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, override

import polars as pl

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

    def collect(
        self,
        n: int | None = None,
        *,
        progress_bar: bool | dict[str, Any] = True,
    ) -> Dataset:
        from flowcean.environments.dataset import collect

        return collect(self, n, progress_bar=progress_bar)

    def num_steps(self) -> int | None:
        return None

    def with_transform(
        self,
        transform: Transform,
    ) -> TransformedIncrementalEnvironment:
        return TransformedIncrementalEnvironment(self, transform)

    def __or__(
        self,
        transform: Transform,
    ) -> TransformedIncrementalEnvironment:
        return self.with_transform(transform)


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

    @override
    def with_transform(
        self,
        transform: Transform,
    ) -> TransformedIncrementalEnvironment:
        transform = self.transform.chain(transform)
        return TransformedIncrementalEnvironment(self.environment, transform)
