from __future__ import annotations

from typing import TYPE_CHECKING, override

import polars as pl

from flowcean.core.environment.observable import TransformedObservable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from flowcean.core.environment.chained import ChainedOfflineEnvironments


class OfflineEnvironment(TransformedObservable):
    def __init__(self) -> None:
        super().__init__()

    def join(self, other: OfflineEnvironment) -> JoinedOfflineEnvironment:
        return JoinedOfflineEnvironment([self, other])

    def __and__(self, other: OfflineEnvironment) -> JoinedOfflineEnvironment:
        return self.join(other)

    def chain(self, *other: OfflineEnvironment) -> ChainedOfflineEnvironments:
        from flowcean.core.environment.chained import (
            ChainedOfflineEnvironments,
        )

        return ChainedOfflineEnvironments([self, *other])

    def __add__(self, other: OfflineEnvironment) -> ChainedOfflineEnvironments:
        return self.chain(other)


class JoinedOfflineEnvironment(OfflineEnvironment):
    environments: Iterable[OfflineEnvironment]

    def __init__(self, environments: Iterable[OfflineEnvironment]) -> None:
        self.environments = environments

    @override
    def _observe(self) -> pl.DataFrame:
        return pl.concat(
            (environment.observe() for environment in self.environments),
            how="horizontal",
        )
