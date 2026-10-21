from abc import ABCMeta, abstractmethod
from typing import Protocol


class Actable[Action](Protocol, metaclass=ABCMeta):
    """A consumer of actions."""

    @abstractmethod
    def act(self, action: Action) -> None:
        """Apply an action to the environment."""
