from __future__ import annotations

from abc import ABC
from typing import override

from flowcean.core.environment.actable import Actable
from flowcean.core.environment.base import Environment
from flowcean.core.environment.incremental import Stepable
from flowcean.core.transform import Transform


class ActiveEnvironment[Observation, Action](
    Environment[Observation],
    Stepable,
    Actable[Action],
    ABC,
):
    """An interactive environment with distinct observation and action types."""

    @override
    def __or__[Result](
        self,
        transform: Transform[Observation, Result],
    ) -> ActiveEnvironment[Result, Action]:
        return _TransformedActiveEnvironment(self, transform)


class _TransformedActiveEnvironment[Input, Output, Action](
    ActiveEnvironment[Output, Action]
):
    def __init__(
        self,
        source: ActiveEnvironment[Input, Action],
        transform: Transform[Input, Output],
    ) -> None:
        self.source = source
        self.observation_transform = transform

    @override
    def _observe(self) -> Output:
        return self.observation_transform(self.source.observe())

    @override
    def act(self, action: Action) -> None:
        self.source.act(action)

    @override
    def step(self) -> None:
        self.source.step()
