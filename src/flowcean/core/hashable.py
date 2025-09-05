from abc import abstractmethod
from typing import Protocol


class Hashable(Protocol):
    """Protocol for hashable objects."""

    @abstractmethod
    def hash(self) -> bytes:
        """Return a hash representing the object state."""
