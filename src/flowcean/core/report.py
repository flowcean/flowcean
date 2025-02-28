from abc import abstractmethod
from typing import Protocol


class Reportable(Protocol):
    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation."""
        raise NotImplementedError


class Report:
    """A report containing reportables."""

    def __init__(self, entries: dict[str, Reportable]) -> None:
        """Initialize the report.

        Args:
            entries: The report entries.
        """
        self.entries = entries

    def __str__(self) -> str:
        """Return a string representation of the report."""
        return "\n".join(
            f"{name}: {value}" for name, value in self.entries.items()
        )
