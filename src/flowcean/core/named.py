from typing import Protocol


class Named(Protocol):
    """A mixin for named objects."""

    _name: str | None = None

    @property
    def name(self) -> str:
        """The name of the object."""
        return self._name or self.__class__.__name__

    @name.setter
    def name(self, value: str) -> None:
        self._name = value
