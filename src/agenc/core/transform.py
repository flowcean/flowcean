"""This module contains the base class for all transforms.

Transforms are used to transform the data before it is used in the learner.
This is useful for scaling the data, or for adding new features to the data.
"""

from abc import ABC, abstractmethod

import polars as pl


class Transform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform the data.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """
