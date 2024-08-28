from __future__ import annotations

from typing import TYPE_CHECKING, override

import polars as pl

from flowcean.core.environment.observable import Observable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from flowcean.core.environment.chained import ChainedOfflineEnvironments
    from flowcean.core.transform import Transform


class OfflineEnvironment(Observable[pl.DataFrame]):
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

    def with_transform(
        self,
        transform: Transform,
    ) -> TransformedOfflineEnvironment:
        return TransformedOfflineEnvironment(self, transform)

    def __or__(
        self,
        transform: Transform,
    ) -> TransformedOfflineEnvironment:
        return self.with_transform(transform)


class JoinedOfflineEnvironment(OfflineEnvironment):
    environments: Iterable[OfflineEnvironment]

    def __init__(self, environments: Iterable[OfflineEnvironment]) -> None:
        self.environments = environments

    @override
    def observe(self) -> pl.DataFrame:
        return pl.concat(
            [environment.observe() for environment in self.environments],
            how="horizontal",
        )


class TransformedOfflineEnvironment(OfflineEnvironment):
    environment: OfflineEnvironment
    transform: Transform

    def __init__(
        self,
        environment: OfflineEnvironment,
        transform: Transform,
    ) -> None:
        self.environment = environment
        self.transform = transform

    @override
    def observe(self) -> pl.DataFrame:
        data = self.environment.observe()
        return self.transform(data)

    @override
    def with_transform(
        self,
        transform: Transform,
    ) -> TransformedOfflineEnvironment:
        transform = self.transform.chain(transform)
        return TransformedOfflineEnvironment(self.environment, transform)
