from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Protocol, override, runtime_checkable

from flowcean.core.environment.base import Environment
from flowcean.core.transform import Transform


@runtime_checkable
class Stepable(Protocol, metaclass=ABCMeta):
    """An environment that can advance by one step."""

    @abstractmethod
    def step(self) -> None:
        """Advance, raising Finished when no observations remain."""


class Finished(Exception):
    """The environment has no further observations."""


class IncrementalEnvironment[Observation](
    Environment[Observation], Stepable, ABC
):
    """An observation stream, yielding the current observation before stepping."""

    _exhausted: bool = False

    @property
    def _finished(self) -> bool:
        return self._exhausted

    @_finished.setter
    def _finished(self, value: bool) -> None:
        self._exhausted = value

    @override
    def observe(self) -> Observation:
        if self._finished:
            raise Finished
        try:
            return super().observe()
        except Finished:
            self._finished = True
            raise

    def __iter__(self) -> Iterator[Observation]:
        try:
            yield self.observe()
            while True:
                self.step()
                yield self.observe()
        except Finished:
            self._finished = True

    def num_steps(self) -> int | None:
        """Return the number of steps, if known."""
        return None

    @override
    def __or__[Result](
        self,
        transform: Transform[Observation, Result],
    ) -> IncrementalEnvironment[Result]:
        return _TransformedIncrementalEnvironment(self, transform)


class _TransformedIncrementalEnvironment[Input, Output](
    IncrementalEnvironment[Output]
):
    def __init__(
        self,
        source: IncrementalEnvironment[Input],
        transform: Transform[Input, Output],
    ) -> None:
        self.source = source
        self.observation_transform = transform

    @property
    @override
    def _finished(self) -> bool:
        return self.source._finished

    @_finished.setter
    def _finished(self, value: bool) -> None:
        self.source._finished = value

    @override
    def _observe(self) -> Output:
        return self.observation_transform(self.source.observe())

    @override
    def step(self) -> None:
        try:
            self.source.step()
        except Finished:
            self.source._finished = True
            raise

    @override
    def num_steps(self) -> int | None:
        return self.source.num_steps()
