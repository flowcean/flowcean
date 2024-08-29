from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, override

import polars as pl

from flowcean.core.environment.observable import (
    TransformedObservable,
)
from flowcean.core.environment.stepable import Finished, Stepable

if TYPE_CHECKING:
    from flowcean.environments.dataset import Dataset


class IncrementalEnvironment(
    TransformedObservable,
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
