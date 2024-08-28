from __future__ import annotations

from abc import ABC, abstractmethod


class Actable[Action](ABC):
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
