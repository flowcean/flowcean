from typing import Any


class Report:
    """A report containing metrics."""

    def __init__(self, metrics: dict[str, Any]) -> None:
        """Initialize the report.

        Args:
            metrics: The metrics in the report.
        """
        self.metrics = metrics

    def __str__(self) -> str:
        """Return the string representation of the report."""
        return "\n".join(
            f"{name}: {value}" for name, value in self.metrics.items()
        )
