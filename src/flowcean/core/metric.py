from abc import ABC, abstractmethod
from typing import Protocol, final

from .named import Named
from .report import Reportable


class Metric[Input, Prepared, Result](Named, ABC):
    """Prepare observations and compute a precisely typed metric result."""

    @abstractmethod
    def prepare(self, data: Input) -> Prepared:
        """Normalize data for the metric implementation."""

    @abstractmethod
    def _compute(self, true: Prepared, predicted: Prepared) -> Result:
        """Calculate the result on prepared inputs."""

    def compute(self, true: Input, predicted: Input) -> Result:
        """Prepare inputs and calculate the result."""
        return self._compute(self.prepare(true), self.prepare(predicted))

    @final
    def __call__(self, true: Input, predicted: Input) -> Result:
        return self.compute(true, predicted)


class MetricCallable[Input, Result](Protocol):
    """The named callable consumed by evaluation and scalar optimization."""

    @property
    def name(self) -> str: ...

    def __call__(self, true: Input, predicted: Input) -> Result: ...


class ActiveMetric[Observation, Action](Named, ABC):
    """Evaluate observations and the actions that produced them."""

    @abstractmethod
    def __call__(
        self,
        observations: list[Observation],
        actions: list[Action],
    ) -> Reportable | dict[str, Reportable]:
        """Calculate a reportable active evaluation result."""
