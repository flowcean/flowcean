from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

from .base import Environment

Action = TypeVar("Action")
Observation = TypeVar("Observation")


class ActiveEnvironment(Environment, Generic[Action, Observation]):
    """Base class for active environments.

    An active environment loads data in an interactive way, e.g., from a
    simulation or real system. The environment requires actions to be taken to
    advance. Data can be retrieved by observing the environment.
    """

    @abstractmethod
    def act(self, action: Action) -> None:
        """Act on the environment.

        Args:
            action: The action to perform.
        """

    @abstractmethod
    def step(self) -> None:
        """Advance the environment by one step."""

    @abstractmethod
    def observe(self) -> Observation:
        """Observe the environment.

        Returns:
            The observation of the environment.
        """
