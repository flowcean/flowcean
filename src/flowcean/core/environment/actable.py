from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Action = TypeVar("Action")


class Actable(Generic[Action], ABC):
    """Base class for active environments.

    Active environments require actions to be taken to advance.
    """

    @abstractmethod
    def act(self, action: Action) -> None:
        """Act on the environment.

        Args:
            action: The action to perform.
        """
