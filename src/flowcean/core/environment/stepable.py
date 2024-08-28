from __future__ import annotations

from abc import ABC, abstractmethod


class Stepable(ABC):
    """Base class for incremental environments.

    An incremental environment loads data in a semi-interactive way, e.g.,
    stream, a sensor, etc. Data can be retrieved multiple times, but the
    environment cannot be controlled.
    """

    @abstractmethod
    def step(self) -> None:
        """Advance the environment by one step."""


class Finished(Exception):
    pass
