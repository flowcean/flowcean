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

    def __or__(self, other: "Transform") -> "Compose":
        """Pipe this transform into another transform.

        Args:
            other: The transform to pipe into.

        Returns:
            A new Compose transform.
        """
        return Compose(self, other)

    def __ror__(self, other: "Transform") -> "Compose":
        """Pipe another transform into this transform.

        Args:
            other: The transform to pipe into this transform.

        Returns:
            A new Compose transform.
        """
        return Compose(other, self)


class Compose(Transform):
    """A transform that is a chain of other transforms."""

    def __init__(self, *transforms: Transform) -> None:
        """Initialize a Compose transform.

        Args:
            transforms: The transforms to pipe together.
        """
        self.transforms = transforms

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        for transform in self.transforms:
            data = transform(data)
        return data

    def __or__(self, other: Transform) -> "Compose":
        return Compose(*self.transforms, other)

    def __ror__(self, other: Transform) -> "Compose":
        return Compose(other, *self.transforms)
