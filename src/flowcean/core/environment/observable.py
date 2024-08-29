from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import polars as pl

from flowcean.core.transform import Identity, Transform


class Observable[Observation](ABC):
    """Base class for observations."""

    @abstractmethod
    def observe(self) -> Observation:
        pass


class TransformedObservable(Observable[pl.DataFrame]):
    transform: Transform

    def __init__(self) -> None:
        super().__init__()
        self.transform = Identity()

    def with_transform(
        self,
        transform: Transform,
    ) -> Self:
        self.transform = self.transform.chain(transform)
        return self

    @abstractmethod
    def _observe(self) -> pl.DataFrame:
        pass

    def observe(self) -> pl.DataFrame:
        return self.transform(self._observe())

    def __or__(
        self,
        transform: Transform,
    ) -> Self:
        return self.with_transform(transform)
