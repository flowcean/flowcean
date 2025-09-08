from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from flowcean.core.data import Data


class Actable(Protocol):
    """Base class for active environments.

    Active environments require actions to be taken to advance.
    """

    @abstractmethod
    def act(self, action: Data) -> None:
        """Act on the environment.

        Args:
            action: The action to perform.
        """
