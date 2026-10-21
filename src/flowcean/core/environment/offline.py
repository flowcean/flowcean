from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from typing import override

from flowcean.core.environment.base import Environment
from flowcean.core.environment.incremental import (
    Finished,
    IncrementalEnvironment,
)
from flowcean.core.transform import Transform


class OfflineEnvironment[Observation](Environment[Observation], ABC):
    """A fixed dataset, composable into an incremental stream."""

    def chain(
        self,
        *other: Environment[Observation],
    ) -> ChainedOfflineEnvironments[Observation]:
        return ChainedOfflineEnvironments([self, *other])

    def __add__(
        self,
        other: Environment[Observation],
    ) -> ChainedOfflineEnvironments[Observation]:
        return self.chain(other)

    @override
    def __or__[Result](
        self,
        transform: Transform[Observation, Result],
    ) -> OfflineEnvironment[Result]:
        return _TransformedOfflineEnvironment(self, transform)


class _TransformedOfflineEnvironment[Input, Output](
    OfflineEnvironment[Output]
):
    def __init__(
        self,
        source: OfflineEnvironment[Input],
        transform: Transform[Input, Output],
    ) -> None:
        self.source = source
        self.observation_transform = transform

    @override
    def _observe(self) -> Output:
        return self.observation_transform(self.source.observe())


class ChainedOfflineEnvironments[Observation](
    IncrementalEnvironment[Observation]
):
    """Observe each source in order, applying its transforms exactly once."""

    def __init__(
        self, environments: Iterable[Environment[Observation]]
    ) -> None:
        self._environments = iter(environments)
        self._element: Environment[Observation] | None
        try:
            self._element = next(self._environments)
        except StopIteration as error:
            message = "environments must contain at least one element"
            raise ValueError(message) from error

    @override
    def _observe(self) -> Observation:
        if self._element is None:
            raise Finished
        return self._element.observe()

    @override
    def step(self) -> None:
        if self._element is None:
            raise Finished
        try:
            self._element = next(self._environments)
        except StopIteration as error:
            self._element = None
            raise Finished from error
