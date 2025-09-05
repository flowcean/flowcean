from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

from typing_extensions import override

from flowcean.core.environment.base import Environment
from flowcean.core.environment.incremental import (
    Finished,
    IncrementalEnvironment,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from flowcean.core.data import Data


@runtime_checkable
class OfflineEnvironment(Environment, Protocol):
    """Base class for offline environments.

    Offline environments are used to represent datasets. They can be used to
    represent static datasets. Offline environments can be transformed and
    joined together to create new datasets.
    """

    def chain(self, *other: Environment) -> ChainedOfflineEnvironments:
        """Chain this offline environment with other offline environments.

        Chaining offline environments will create a new incremental environment
        that will first observe the data from this environment and then the
        data from the other environments.

        Args:
            other: The other offline environments to chain.

        Returns:
            The chained offline environments.
        """
        return ChainedOfflineEnvironments([self, *other])

    @final
    def __add__(self, other: Environment) -> ChainedOfflineEnvironments:
        """Shorthand for `chain`."""
        return self.chain(other)


class ChainedOfflineEnvironments(IncrementalEnvironment):
    """Chained offline environments.

    This environment chains multiple offline environments together. The
    environment will first observe the data from the first environment and then
    the data from the other environments.
    """

    _environments: Iterator[Environment]
    _element: Environment

    def __init__(self, environments: Iterable[Environment]) -> None:
        """Initialize the chained offline environments.

        Args:
            environments: The offline environments to chain.
        """
        self._environments = iter(environments)
        self._element = next(self._environments)

    @override
    def _observe(self) -> Data:
        return self._element.observe()

    @override
    def step(self) -> None:
        try:
            self._element = next(self._environments)
        except StopIteration:
            raise Finished from StopIteration
