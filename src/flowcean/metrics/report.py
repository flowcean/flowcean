from typing import Any


class Report:
    def __init__(self, metrics: dict[str, Any]) -> None:
        self.metrics = metrics

    def __str__(self) -> str:
        return "\n".join(
            f"{name}: {value}" for name, value in self.metrics.items()
        )
