from __future__ import annotations

from typing import TYPE_CHECKING

from flowcean.core.environment.observable import TransformedObservable

if TYPE_CHECKING:
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
