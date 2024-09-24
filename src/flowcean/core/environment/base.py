from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from flowcean.core import Transform

    from .transformed import TransformedEnvironment


class Environment(ABC):
    """Base class for environments.

    An environment provides data to a learner. It can be a data loader, a
    simulation, etc.
    """

    @abstractmethod
    def load(self) -> Self:
        """Load the environment.

        This method must be called before accessing the data of the
        environment. Environments use this method to initialize themselves and
        load their data. An environment should be loaded only once.

        Returns:
            The loaded environment.
        """

    def with_transform(
        self,
        transform: Transform,
    ) -> TransformedEnvironment[Self]:
        """Attach a transform to the environment.

        Attach a transform to the environment by wrapping it in a
        [`TransformedEnvironment`][flowcean.core.environment.TransformedEnvironment]
        instance. The transform will be applied to any data of the environment
        before returning it.

        Args:
            transform: The transform to apply to the data of the environment.

        Returns:
            The transformed environment.
        """
        from .transformed import TransformedEnvironment

        return TransformedEnvironment(self, transform)


class NotLoadedError(Exception):
    """Environment not loaded.

    This exception is raised when trying to access the data of an environment
    that has not been loaded yet.
    """
