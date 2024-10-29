from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, override

import polars as pl

from flowcean.core.transform import Identity, Transform


class Observable[Observation](ABC):
    """Base class for observations."""

    @abstractmethod
    def observe(self) -> Observation:
        """Observe and return the observation."""


class TransformedObservable(Observable[pl.DataFrame]):
    """Base class for observations that carry a transform.

    Attributes:
        transform: Transform
    """

    transform: Transform

    def __init__(self) -> None:
        """Initialize the observable."""
        super().__init__()
        self.transform = Identity()

    def with_transform(
        self,
        transform: Transform,
    ) -> Self:
        """Append a transform to the observation.

        Args:
            transform: Transform to append.

        Returns:
            This observable with the appended transform.
        """
        self.transform = self.transform.chain(transform)
        return self

    @abstractmethod
    def _observe(self) -> pl.DataFrame:
        """Observe and return the observation without applying the transform.

        This method must be implemented by subclasses.

        Returns:
            The raw observation.
        """

    @override
    def observe(self) -> pl.DataFrame:
        return self.transform(self._observe())

    def __or__(
        self,
        transform: Transform,
    ) -> Self:
        """Shortcut for `with_transform`."""
        return self.with_transform(transform)
