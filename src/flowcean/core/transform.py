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
dataset: pl.DataFrame = ...

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
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


class Transform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Apply the transform to data.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        """Apply the transform to data.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """
        return self.transform(data)

    def chain(
        self,
        *other: Transform,
    ) -> Chain:
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
        return Chain(self, *other)

    def __or__(
        self,
        other: Transform,
    ) -> Chain:
        """Chain this transform with another transform.

        This can be used to chain multiple transforms together.
        Chained transforms are applied left to right.

        Example:
            ```python
            chained_transform = TransformA() >> TransformB()
            ```

        Args:
            other: The transform to chain.

        Returns:
            A new Chain transform.
        """
        return self.chain(other)


class Chain(
    Transform,
    # UnsupervisedLearner,
    # UnsupervisedIncrementalLearner,
):
    """A transform that is a chain of other transforms."""

    transforms: Sequence[Transform]

    def __init__(
        self,
        *transforms: Transform,
    ) -> None:
        self.transforms = transforms

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        for transform in self.transforms:
            data = transform.transform(data)
        return data

    # @override
    # def fit(self, data: pl.DataFrame) -> None:
    #     for transform in self.transforms:
    #         if isinstance(transform, UnsupervisedLearner):
    #             transform.fit(data)
    #         data = transform.transform(data)
    #
    # @override
    # def fit_incremental(self, data: pl.DataFrame) -> None:
    #     for transform in self.transforms:
    #         if isinstance(transform, UnsupervisedIncrementalLearner):
    #             transform.fit_incremental(data)
    #         data = transform.transform(data)
