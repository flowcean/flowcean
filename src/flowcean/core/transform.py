"""This module provides base classes for transforms.

Pre-processing of data, feature engineering, or augmentation, are fundamental
processes in machine learning.
AGenC generalizes these processes under the term _transforms_.
This page will guide through the concept of transforms and demonstrate how to
use them within AGenC.

## Nomenclature

Transforms are a set of operations that modify data.
They can include operations such as data normalization, dimensionality
reduction, data augmentation, and much more.
These transformations are essential for preparing data for machine learning
tasks and improving model performance.

In AGenC, we use the generalized term _transform_ for all types of
_pre-processing of data_, _feature engineering_, and _data augmentation_, as
they all involve the same fundamental concept of transforming data to obtain a
modified dataset.

AGenC provides a flexible and unified interface to apply transforms to data.
The framework allows to combine these transforming steps steps as needed.

## Using Transforms

Here's a basic example:

```python
from flowcean.transforms import Select, Standardize

# Load the dataset
dataset = ...

# Define transforms by chaining a selection and a standardization
transforms = Select(features=["reference", "temperature"]) | Standardize(
    mean={
        "reference": 0.0,
        "temperature": 0.0,
    },
    std={
        "reference": 1.0,
        "temperature": 1.0,
    },
)

# Apply the transforms to data
transformed_data = transforms(dataset)
```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flowcean.core.data import Data


class Transform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def apply(self, data: Data) -> Data:
        """Apply the transform to data.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """

    def __call__(self, data: Data) -> Data:
        """Apply the transform to data.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """
        return self.apply(data)

    def chain(
        self,
        other: Transform,
    ) -> Transform:
        """Chain this transform with other transforms.

        This can be used to chain multiple transforms together.
        Chained transforms are applied left to right.

        Example:
            ```python
            chained_transform = TransformA().chain(TransformB())
            ```

        Args:
            other: The transforms to chain.

        Returns:
            A new Chain transform.
        """
        return ChainedTransforms(self, other)

    def __or__(
        self,
        other: Transform,
    ) -> Transform:
        """Shorthand for chaining transforms.

        Example:
            ```python
            chained_transform = TransformA() | TransformB()
            ```

        Args:
            other: The transform to chain.

        Returns:
            A new Chain transform.
        """
        return self.chain(other)

    def inverse(self) -> Transform:
        """Get the inverse of the transform.

        Returns:
            The inverse of the transform.
        """
        raise NotImplementedError


class FitOnce(ABC):
    """A mixin for transforms that need to be fitted to data once."""

    @abstractmethod
    def fit(self, data: Data) -> None:
        """Fit to the data.

        Args:
            data: The data to fit to.
        """


class FitIncremetally(ABC):
    """A mixin for transforms that need to be fitted to data incrementally."""

    @abstractmethod
    def fit_incremental(self, data: Data) -> None:
        """Fit to the data incrementally.

        Args:
            data: The data to fit to.
        """


class ChainedTransforms(Transform, FitOnce, FitIncremetally):
    """A transform that is a chain of other transforms."""

    transforms: Sequence[Transform]

    def __init__(
        self,
        *transforms: Transform,
    ) -> None:
        """Initialize the chained transforms.

        Args:
            transforms: The transforms to chain.
        """
        self.transforms = transforms

    @override
    def apply(self, data: Data) -> Data:
        for transform in self.transforms:
            data = transform.apply(data)
        return data

    @override
    def chain(
        self,
        other: Transform,
    ) -> Transform:
        return ChainedTransforms(*self.transforms, other)

    @override
    def fit(self, data: Data) -> None:
        for transform in self.transforms:
            if isinstance(transform, FitOnce):
                transform.fit(data)
            data = transform.apply(data)

    @override
    def fit_incremental(self, data: Data) -> None:
        for transform in self.transforms:
            if isinstance(transform, FitIncremetally):
                transform.fit_incremental(data)
            data = transform.apply(data)


class Identity(Transform):
    """A transform that does nothing."""

    def __init__(self) -> None:
        """Initialize the identity transform."""
        super().__init__()

    @override
    def apply(self, data: Data) -> Data:
        return data

    @override
    def chain(
        self,
        other: Transform,
    ) -> Transform:
        return other
