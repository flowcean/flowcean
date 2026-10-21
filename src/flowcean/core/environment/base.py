from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, override

from flowcean.core.named import Named
from flowcean.core.transform import Identity, Transform


class Environment[Observation](Named, ABC):
    """A source of observations with type-preserving mutable transforms."""

    @property
    def transform(self) -> Transform[Observation, Observation]:
        if "_transform" not in self.__dict__:
            self._transform: Transform[Observation, Observation] = Identity()
        return self._transform

    @transform.setter
    def transform(
        self, transform: Transform[Observation, Observation]
    ) -> None:
        self._transform = transform

    def append_transform(
        self,
        transform: Transform[Observation, Observation],
    ) -> Self:
        """Mutate this environment without changing its observation type."""
        self.transform = self.transform | transform
        return self

    @abstractmethod
    def _observe(self) -> Observation:
        """Return an untransformed observation."""

    def observe(self) -> Observation:
        """Observe, then apply this environment's transforms."""
        return self.transform(self._observe())

    def __or__[Result](
        self,
        transform: Transform[Observation, Result],
    ) -> Environment[Result]:
        """Create a new view, optionally changing the observation type."""
        return _TransformedEnvironment(self, transform)


class _TransformedEnvironment[Input, Output](Environment[Output]):
    def __init__(
        self,
        source: Environment[Input],
        transform: Transform[Input, Output],
    ) -> None:
        self.source = source
        self.observation_transform = transform

    @override
    def _observe(self) -> Output:
        return self.observation_transform(self.source.observe())
