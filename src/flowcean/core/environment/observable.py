from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

from typing_extensions import Self, override

from flowcean.core.transform import Identity, Transform

if TYPE_CHECKING:
    from flowcean.core.data import Data


class NotSupportedError(Exception):
    """Error raised when hashing is not supported by the observable."""

    def __init__(self, observable: Observable) -> None:
        """Initialize the error.

        Args:
            observable: Observable that does not support hashing.
        """
        super().__init__(f"Observable {observable} does not support hashing.")


class Observable(Protocol):
    """Protocol for observations."""

    @abstractmethod
    def observe(self) -> Data:
        """Observe and return the observation."""
        raise NotImplementedError

    @abstractmethod
    def hash(self) -> bytes:
        """Return a hash of the observable.

        Raises:
            NotSupportedError: If the observable does not support hashing.
        """


class TransformedObservable(Observable):
    """Base class for observations that carry a transform.

    Attributes:
        transform: Transform
    """

    transform: Transform

    def __init__(self) -> None:
        """Initialize the observable."""
        super().__init__()
        self.transform = Identity()

    def with_transform(
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

    @override
    def observe(self) -> Data:
        return self.transform(self._observe())

    def __or__(
        self,
        transform: Transform,
    ) -> Self:
        """Shortcut for `with_transform`."""
        return self.with_transform(transform)
