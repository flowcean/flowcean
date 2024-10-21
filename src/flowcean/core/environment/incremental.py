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
    """Base class for incremental environments.

    Incremental environments are environments that can be advanced by a step
    and provide a stream of data. The data can be observed at each step.
    """

    def __init__(self) -> None:
        """Initialize the incremental environment."""
        super().__init__()

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
        """Collect data from the environment.

        Args:
            n: Number of steps to collect. If None, all steps are collected.
            progress_bar: Whether to show a progress bar. If a dictionary is
                provided, it will be passed to the progress bar.

        Returns:
            The collected dataset.
        """
        from flowcean.environments.dataset import collect

        return collect(self, n, progress_bar=progress_bar)

    def num_steps(self) -> int | None:
        """Return the number of steps in the environment.

        Returns:
            The number of steps in the environment, or None if the number of
            steps is unknown.
        """
        return None
