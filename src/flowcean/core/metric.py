from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Protocol,
)

if TYPE_CHECKING:
    from .data import Data
    from .report import Reportable


class SupportsPrepare(Protocol):
    """Protocol describing an object that has a `prepare` method."""

    def prepare(self, data: Data) -> Data: ...


class SupportsCompute(Protocol):
    """Protocol describing an object that has a `compute` method."""

    def compute(self, true: Data, predicted: Data) -> Reportable: ...


class OfflineMetric(ABC):
    """Minimal template for offline metrics.

    Call flow:
      __call__ -> prepare(true), prepare(predicted) -> compute(true, predicted)
    """

    def __init__(self, *, name: str | None = None) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    def __call__(
        self,
        true: Data,
        predicted: Data,
    ) -> Reportable | dict[str, Reportable]:
        """Execute metric: prepare inputs then compute."""
        t = self.prepare(true)
        p = self.prepare(predicted)
        return self.compute(t, p)

    def prepare(self, data: Data) -> Data:
        """Hook to normalize/collect/select data before computing metric.

        Default: identity. Mixins override and call super().prepare(...)
        """
        return data

    @abstractmethod
    def compute(self, true: Data, predicted: Data) -> Reportable:
        """Implement metric logic on prepared inputs."""
