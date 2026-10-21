"""Typed, composable transforms and explicit optional capabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    Protocol,
    Self,
    TypeGuard,
    final,
    override,
    runtime_checkable,
)

import cloudpickle

from .named import Named


class Transform[Input, Output](Named, ABC):
    """A reusable operation, with optional batch fitting.

    Composition is binary and applies left to right. Use inverse() and
    fit_incremental() functions to check optional capabilities across a chain.
    """

    @abstractmethod
    def apply(self, data: Input) -> Output:
        """Transform data."""

    @final
    def __call__(self, data: Input) -> Output:
        return self.apply(data)

    def chain[Result](
        self,
        other: Transform[Output, Result],
    ) -> ChainedTransforms[Input, Output, Result]:
        """Create a new composition, without mutating either transform."""
        return ChainedTransforms(self, other)

    def __or__[Result](
        self,
        other: Transform[Output, Result],
    ) -> ChainedTransforms[Input, Output, Result]:
        return self.chain(other)

    def fit(self, data: Input) -> Self:
        """Fit to a batch. The default implementation is a no-op."""
        return self


@runtime_checkable
class Invertible[Input, Output](Protocol):
    """Guarantees an inverse operation; fitting may still be required."""

    def inverse(self) -> Transform[Output, Input]:
        """Return the inverse transform."""
        ...


@runtime_checkable
class IncrementallyFittable[Input](Protocol):
    """A transform that supports learning from successive batches."""

    def fit_incremental(self, data: Input) -> Self: ...


class InvertibleTransform[Input, Output](Transform[Input, Output], ABC):
    """A transform with a guaranteed inverse operation."""

    @abstractmethod
    def inverse(self) -> Transform[Output, Input]:
        """Return the inverse transform, after fitting if necessary."""


class StatelessTransform[Input, Output](Transform[Input, Output], ABC):
    """A transform requiring no fitting, including on incremental batches."""

    def fit_incremental(self, data: Input) -> Self:
        """No state needs to be fitted."""
        return self


class ChainedTransforms[Input, Intermediate, Output](Transform[Input, Output]):
    """Two transforms applied and fitted in observation order."""

    def __init__(
        self,
        first: Transform[Input, Intermediate],
        second: Transform[Intermediate, Output],
    ) -> None:
        self.first = first
        self.second = second

    @override
    def apply(self, data: Input) -> Output:
        return self.second(self.first(data))

    @override
    def fit(self, data: Input) -> Self:
        self.first.fit(data)
        self.second.fit(self.first(data))
        return self

    def inverse_chain(self) -> Transform[Output, Input]:
        return inverse(self.second) | inverse(self.first)

    def validate_incremental_fit(self) -> None:
        validate_incremental_fit(self.first)
        validate_incremental_fit(self.second)

    def fit_incremental_chain(self, data: Input) -> None:
        self.validate_incremental_fit()
        fit_incremental(self.first, data)
        fit_incremental(self.second, self.first(data))


def inverse[Input, Output](
    transform: Transform[Input, Output],
) -> Transform[Output, Input]:
    """Resolve an inverse, checking every leaf of a composition.

    General chains do not satisfy Invertible: this checked operation rejects
    unsupported leaves and unfitted or noninvertible fitted state.
    """
    if isinstance(transform, ChainedTransforms):
        return transform.inverse_chain()
    if _is_invertible(transform):
        return transform.inverse()
    message = f"{transform.name} does not support inversion"
    raise TypeError(message)


def validate_incremental_fit[Input, Output](
    transform: Transform[Input, Output],
) -> None:
    """Reject unsupported leaves before any transform or learner mutation."""
    if isinstance(transform, ChainedTransforms):
        transform.validate_incremental_fit()
    elif not isinstance(transform, IncrementallyFittable):
        message = f"{transform.name} does not support incremental fitting"
        raise TypeError(message)


def fit_incremental[Input, Output](
    transform: Transform[Input, Output],
    data: Input,
) -> None:
    """Validate the whole pipeline, then fit in application order."""
    validate_incremental_fit(transform)
    if isinstance(transform, ChainedTransforms):
        transform.fit_incremental_chain(data)
    elif _is_incrementally_fittable(transform):
        transform.fit_incremental(data)


def _is_invertible[Input, Output](
    transform: Transform[Input, Output],
) -> TypeGuard[Invertible[Input, Output]]:
    # Capability type parameters follow the transform's input/output contract.
    return isinstance(transform, Invertible)


def _is_incrementally_fittable[Input, Output](
    transform: Transform[Input, Output],
) -> TypeGuard[IncrementallyFittable[Input]]:
    return isinstance(transform, IncrementallyFittable)


class Identity[Data](
    StatelessTransform[Data, Data], InvertibleTransform[Data, Data]
):
    """Return data unchanged."""

    @override
    def apply(self, data: Data) -> Data:
        return data

    @override
    def inverse(self) -> Identity[Data]:
        return self


class Lambda[Input, Output](StatelessTransform[Input, Output]):
    """Wrap a function. Use InvertibleLambda when an inverse is available."""

    def __init__(self, func: Callable[[Input], Output]) -> None:
        self.func = func

    @override
    def apply(self, data: Input) -> Output:
        return self.func(data)

    def __getstate__(self) -> dict[str, bytes]:
        return {"func": cloudpickle.dumps(self.func)}

    def __setstate__(self, state: dict[str, bytes]) -> None:
        self.func = cloudpickle.loads(state["func"])


class InvertibleLambda[Input, Output](
    Lambda[Input, Output],
    InvertibleTransform[Input, Output],
):
    """Wrap a function and its inverse, preserving both during serialization."""

    def __init__(
        self,
        func: Callable[[Input], Output],
        inverse_func: Callable[[Output], Input],
    ) -> None:
        super().__init__(func)
        self.inverse_func = inverse_func

    @override
    def inverse(self) -> InvertibleLambda[Output, Input]:
        return InvertibleLambda(self.inverse_func, self.func)

    @override
    def __getstate__(self) -> dict[str, bytes]:
        return {
            "func": cloudpickle.dumps(self.func),
            "inverse_func": cloudpickle.dumps(self.inverse_func),
        }

    @override
    def __setstate__(self, state: dict[str, bytes]) -> None:
        super().__setstate__(state)
        self.inverse_func = cloudpickle.loads(state["inverse_func"])
