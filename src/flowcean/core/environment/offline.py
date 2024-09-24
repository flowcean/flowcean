from __future__ import annotations

from typing import TYPE_CHECKING, override

import polars as pl

from flowcean.core.environment.observable import TransformedObservable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from flowcean.core.environment.chained import ChainedOfflineEnvironments


class OfflineEnvironment(TransformedObservable):
    """Base class for offline environments.

    Offline environments are used to represent datasets. They can be used to
    represent static datasets. Offline environments can be transformed and
    joined together to create new datasets.
    """

    def __init__(self) -> None:
        """Initialize the offline environment."""
        super().__init__()

    def join(self, other: OfflineEnvironment) -> JoinedOfflineEnvironment:
        """Join this offline environment with another one.

        Args:
            other: The other offline environment to join.

        Returns:
            The joined offline environment.
        """
        return JoinedOfflineEnvironment([self, other])

    def __and__(self, other: OfflineEnvironment) -> JoinedOfflineEnvironment:
        """Shorthand for `join`."""
        return self.join(other)

    def chain(self, *other: OfflineEnvironment) -> ChainedOfflineEnvironments:
        """Chain this offline environment with other offline environments.

        Chaining offline environments will create a new incremental environment
        that will first observe the data from this environment and then the
        data from the other environments.

        Args:
            other: The other offline environments to chain.

        Returns:
            The chained offline environments.
        """
        from flowcean.core.environment.chained import (
            ChainedOfflineEnvironments,
        )

        return ChainedOfflineEnvironments([self, *other])

    def __add__(self, other: OfflineEnvironment) -> ChainedOfflineEnvironments:
        """Shorthand for `chain`."""
        return self.chain(other)


class JoinedOfflineEnvironment(OfflineEnvironment):
    """Environment that joins multiple offline environments.

    Attributes:
        environments: The offline environments to join.
    """

    environments: Iterable[OfflineEnvironment]

    def __init__(self, environments: Iterable[OfflineEnvironment]) -> None:
        """Initialize the joined offline environment.

        Args:
            environments: The offline environments to join.
        """
        self.environments = environments
        super().__init__()

    @override
    def _observe(self) -> pl.DataFrame:
        return pl.concat(
            (environment.observe() for environment in self.environments),
            how="horizontal",
        )
