from __future__ import annotations

from abc import ABC, abstractmethod


class Observable[Observation](ABC):
    """Base class for observations."""

    @abstractmethod
    def observe(self) -> Observation:
        pass
