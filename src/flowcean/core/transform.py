"""This module provides abstractions for transforms.

Transforms are reusable, composable operations that modify data in preparation
for machine learning or analysis. Transforms unify pre-processing, feature
engineering, and augmentation under a single protocol-based interface.

## Nomenclature

Transforms are a set of operations that modify data. They can include
operations such as data normalization, dimensionality reduction, data
augmentation, and much more. These transformations are essential for preparing
data for machine learning tasks and improving model performance.

We use the generalized term _transform_ for all types of _pre-processing of
data_, _feature engineering_, and _data augmentation_, as they all involve the
same fundamental concept of transforming data to obtain a modified dataset.

flowcean provides a flexible and unified interface to apply transforms to data.
The framework allows to combine these transforming steps steps as needed.

## Using Transforms

Here's a basic example:

```python
from flowcean.polars import Select, Standardize
from flowcean.core import Lambda

# Define a simple pipeline
transform = (
    Select(features=["x", "y"])
    | Standardize()
    | Lambda(lambda df: df.with_columns(z=df["x"] * df["y"]))
)

# Fit and apply
transform.fit(dataset)
transformed = transform(dataset)

# Invert (if supported)
restored = transform.inverse()(transformed)
```
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, cast, final, runtime_checkable

import cloudpickle
from typing_extensions import Self, override

from .named import Named

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from flowcean.core.data import Data


class Transform(Named, Protocol):
    """Base protocol for all transforms in Flowcean.

    A transform is a reusable operation that modifies data. Examples include
    preprocessing (e.g., standardization), feature engineering (e.g., feature
    selection, PCA), or augmentation (noise injection, synthetic features).

    Transforms are composable via the ``|`` operator, allowing complex
    transformation pipelines to be expressed in a clean and functional style:

    Example:
        ```
        >>> transform = Select(features=["x"]) | Standardize()
        >>> transformed = transform(dataset)
        ```
    """

    @abstractmethod
    def apply(self, data: Data) -> Data:
        """Apply the transform to data.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """

    @final
    def __call__(self, data: Data) -> Data:
        """Apply the transform to data.

        Equivalent to ``self.apply(data)``.

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
        """Chain this transform with ``other``.

        This can be used to chain multiple transforms together. Chained
        transforms are applied left-to-right:

        Example:
            ```python
            chained = TransformA().chain(TransformB())
            chained(data)  # Equivalent to TransformB(TransformA(data))
            ```

        Args:
            other: The transforms to chain.

        Returns:
            A new chained transform.
        """
        return ChainedTransforms(self, other)

    def __or__(
        self,
        other: Transform,
    ) -> Transform:
        """Shorthand for chaining transforms.

        Example:
            ```python
            chained = TransformA() | TransformB()
            ```

        Args:
            other: The transform to chain.

        Returns:
            A new Chain transform.
        """
        return self.chain(other)

    def fit(self, data: Data) -> Self:
        """Fit the transform to data.

        Many transforms (e.g. scaling, PCA) require statistics from the dataset
        before applying. Default implementation is a no-op.
        This is meant to be idempotent, i.e., calling ``fit()`` multiple times
        should have the same effect as calling it once.

        Args:
            data: The data to fit to.
        """
        _ = data
        return self

    def fit_incremental(self, data: Data) -> Self:
        """Incrementally fit the transform to streaming/batched data.

        Default implementation is a no-op.

        Args:
            data: The data to fit to.
        """
        _ = data
        return self


@runtime_checkable
class Invertible(Protocol):
    """Protocol for transforms that support inversion.

    An invertible transform can undo its effect via ``inverse()``.

    Example:
        ```
        >>> scaler = Standardize().fit(data)
        >>> restored = scaler.inverse()(scaler(data))
        ```
    """

    @abstractmethod
    def inverse(self) -> Transform:
        """Return a new transform that inverts this one.

        Returns:
            The inverse of the transform.
        """


class ChainedTransforms(Invertible, Transform):
    """A composition of multiple transforms applied sequentially.

    Chained transforms are applied left-to-right. Useful for building
    preprocessing pipelines.
    """

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
    def fit(self, data: Data) -> Self:
        for transform in self.transforms:
            transform.fit(data)
            data = transform.apply(data)
        return self

    @override
    def fit_incremental(self, data: Data) -> Self:
        for transform in self.transforms:
            transform.fit_incremental(data)
            data = transform.apply(data)
        return self

    @override
    def inverse(self) -> Transform:
        non_inv = [t for t in self.transforms if not isinstance(t, Invertible)]
        if non_inv:
            msg = f"Transforms {non_inv} do not support inversion."
            raise NotImplementedError(msg)
        transforms = cast("Sequence[Invertible]", self.transforms)
        return ChainedTransforms(
            *(t.inverse() for t in reversed(transforms)),
        )


class InvertibleTransform(Transform, Invertible, Protocol): ...


class Identity(Invertible, Transform):
    """A no-op transform that returns data unchanged.

    Often used as a placeholder or default transform.
    """

    def __init__(self) -> None:
        """Initialize the identity transform."""
        super().__init__()

    @override
    def apply(self, data: Data) -> Data:
        return data

    @override
    def inverse(self) -> Transform:
        return self

    @override
    def chain(
        self,
        other: Transform,
    ) -> Transform:
        return other


class Lambda(Transform, Invertible):
    """A transform wrapping a function.

    Useful for quick one-off transformations without creating a dedicated
    class.

    Example:
        ```
        >>> to_float = Lambda(lambda df: df.cast(pl.Float64))
        >>> normalized = Lambda(
        ...     lambda df: (df - df.mean()) / df.std(),
        ...     inverse_func=lambda df: df * df.std() + df.mean(),
        ... )
        ```
    """

    func: Callable[[Data], Data]
    inverse_func: Callable[[Data], Data] | None

    def __init__(
        self,
        func: Callable[[Data], Data],
        *,
        inverse_func: Callable[[Data], Data] | None = None,
    ) -> None:
        """Initialize the lambda transform.

        Args:
            func: Function that transforms data.
            inverse_func: Optional function that inverts ``func``.
        """
        self.func = func
        self.inverse_func = inverse_func

    @override
    def apply(self, data: Data) -> Data:
        return self.func(data)

    def __getstate__(self) -> dict[str, bytes]:
        return {"func": cloudpickle.dumps(self.func)}

    def __setstate__(self, state: dict[str, bytes]) -> None:
        self.func = cloudpickle.loads(state["func"])

    def inverse(self) -> Transform:
        if self.inverse_func is None:
            msg = "This transform does not have an inverse."
            raise NotImplementedError(msg)
        return Lambda(self.inverse_func, inverse_func=self.func)
