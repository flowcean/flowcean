from __future__ import annotations

from typing import TYPE_CHECKING, override

from flowcean.core.environment.observable import Observable

if TYPE_CHECKING:
    from flowcean.core.transform import Transform


class Environment[Observation, Out](Observable[Observation]):
    transform: Transform[Observation, Out]

    def __init__(self, transform: Transform[Observation, Out]) -> None:
        self.transform = transform

    @override
    def observe(self) -> Out:
        observation = super().observe()
        return self.transform(observation)
