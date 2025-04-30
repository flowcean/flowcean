from __future__ import annotations

from abc import abstractmethod
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

    @abstractmethod
    def hash(self) -> bytes:
        """Return the hash of the offline environment.

        The hash of the offline environment is used to uniquely identify the
        data of an environment, mainly for caching purposes. The following
        properties should be considered when computing the hash:

        - If two environments are equal (e.g. are the same object),
          they must have the same hash.
        - If two environments of the same type have the same hash, their data
          must be equal.
        - If two environments have different data, their hashes must be
          different.

        These properties leave one special case open: If two environments share
        the same data, the hash may be different even if the data is not.
        The hash is just a way of quickly checking if there is a chance that
        the data is different between two environments.

        Returns:
            The hash of the offline environment.
        """
