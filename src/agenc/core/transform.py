from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from .chain import Chain


class Transform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform the data.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """

    def __or__(self, other: Transform) -> Chain:
        """Pipe this transform into another transform.

        Args:
            other: The transform to pipe into.

        Returns:
            A new Chain transform.
        """
        from .chain import Chain

        return Chain(self, other)
