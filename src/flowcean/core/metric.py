from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Protocol,
    final,
)

from flowcean.core.named import Named

if TYPE_CHECKING:
    from .data import Data
    from .report import Reportable


class Metric(Named, Protocol):
    """Minimal template for metrics.

    Call flow:
      __call__ -> prepare(true), prepare(predicted) -> compute(true, predicted)
    """

    def prepare(self, data: Data) -> Data:
        """Hook to normalize/collect/select data before computing metric.

        Default: identity. Mixins override and call super().prepare(...)
        """
        return data

    @abstractmethod
    def _compute(self, true: Data, predicted: Data) -> Reportable:
        """Implement metric logic on prepared inputs."""

    @final
    def __call__(
        self,
        true: Data,
        predicted: Data,
    ) -> Reportable | dict[str, Reportable]:
        """Execute metric: prepare inputs then compute."""
        return self.compute(true, predicted)

    def compute(self, true: Data, predicted: Data) -> Reportable:
        """Implement metric logic on prepared inputs."""
        t = self.prepare(true)
        p = self.prepare(predicted)
        return self._compute(t, p)
