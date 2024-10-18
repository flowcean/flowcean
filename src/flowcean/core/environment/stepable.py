from __future__ import annotations

from abc import ABC, abstractmethod


class Stepable(ABC):
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
