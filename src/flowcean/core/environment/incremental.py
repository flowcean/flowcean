from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from typing_extensions import override

from flowcean.core.data import Data
from flowcean.core.environment.base import Environment


@runtime_checkable
class Stepable(Protocol):
    """Base class for stepable environments.

    Stepable environments are environments that can be advanced by a step.
    Usually, this is combined with an observable to provide a stream of data.
    """

    @abstractmethod
    def step(self) -> None:
        """Advance the environment by one step."""


class Finished(Exception):
    """Exception raised when the environment is finished.

    This exception is raised when the environment is finished, and no more data
    can be retrieved.
    """


@runtime_checkable
class IncrementalEnvironment(
    Environment,
    Stepable,
    Iterable[Data],
    Protocol,
):
    """Base class for incremental environments.

    Incremental environments are environments that can be advanced by a step
    and provide a stream of data. The data can be observed at each step.
    """

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
