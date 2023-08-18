"""
Transforms are used to transform the data before it is used in the learner.
This is useful for scaling the data, or for adding new features to the data.
"""

import polars as pl
from abc import ABC, abstractmethod


class Transform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform the data.

        Args:
            data (pl.DataFrame): The data to transform.

        Returns:
            pl.DataFrame: The transformed data.
        """


class StandardScaler(Transform):
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select((pl.all() - pl.all().mean()) / pl.all().std())
