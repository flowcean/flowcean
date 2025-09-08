from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

from typing_extensions import Self

from flowcean.core.named import Named
from flowcean.core.transform import Identity, Transform

if TYPE_CHECKING:
    from flowcean.core.data import Data


@runtime_checkable
class Environment(Named, Protocol):
    """Base class adding transform support."""

    transform: Transform = Identity()

    def append_transform(
        self,
        transform: Transform,
    ) -> Self:
        """Append a transform to the observation.

        Args:
            transform: Transform to append.

        Returns:
            This observable with the appended transform.
        """
        self.transform = self.transform.chain(transform)
        return self

    @abstractmethod
    def _observe(self) -> Data:
        """Observe and return the observation without applying the transform.

        This method must be implemented by subclasses.

        Returns:
            The raw observation.
        """

    def observe(self) -> Data:
        """Observe and return the observation."""
        return self.transform(self._observe())

    @final
    def __or__(
        self,
        transform: Transform,
    ) -> Self:
        """Shortcut for `with_transform`."""
        return self.append_transform(transform)
