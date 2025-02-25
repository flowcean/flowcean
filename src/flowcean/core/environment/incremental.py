from __future__ import annotations

from collections.abc import Iterable, Iterator

from typing_extensions import override

from flowcean.core import (
    Finished,
    HashingNotSupportedError,
    Stepable,
    TransformedObservable,
)
from flowcean.core.data import Data


class IncrementalEnvironment(
    TransformedObservable,
    Stepable,
    Iterable[Data],
):
    """Base class for incremental environments.

    Incremental environments are environments that can be advanced by a step
    and provide a stream of data. The data can be observed at each step.
    """

    def __init__(self) -> None:
        """Initialize the incremental environment."""
        super().__init__()

    @override
    def hash(self) -> bytes:
        raise HashingNotSupportedError

    @override
    def __iter__(self) -> Iterator[Data]:
        yield self.observe()
        while True:
            try:
                self.step()
            except Finished:
                break
            yield self.observe()

    def num_steps(self) -> int | None:
        """Return the number of steps in the environment.

        Returns:
            The number of steps in the environment, or None if the number of
            steps is unknown.
        """
        return None
