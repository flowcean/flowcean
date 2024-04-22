from __future__ import annotations

from typing import TYPE_CHECKING, override

from flowcean.core.learner import (
    UnsupervisedIncrementalLearner,
    UnsupervisedLearner,
)

from .transform import Transform

if TYPE_CHECKING:
    import polars as pl


class Chain(Transform, UnsupervisedLearner, UnsupervisedIncrementalLearner):
    """A transform that is a chain of other transforms."""

    transforms: tuple[Transform, ...]

    def __init__(
        self,
        *transforms: Transform,
    ) -> None:
        """Initialize a Chain transform.

        Args:
            transforms: The transforms to pipe together.
        """
        self.transforms = transforms

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        for transform in self.transforms:
            data = transform.transform(data)
        return data

    @override
    def __or__(self, other: Transform) -> Chain:
        """Pipe this transform into another transform.

        Args:
            other: The transform to pipe into.
        """
        return Chain(*self.transforms, other)

    @override
    def fit(self, data: pl.DataFrame) -> None:
        for transform in self.transforms:
            if isinstance(transform, UnsupervisedLearner):
                transform.fit(data)
            data = transform.transform(data)

    @override
    def fit_incremental(self, data: pl.DataFrame) -> None:
        for transform in self.transforms:
            if isinstance(transform, UnsupervisedIncrementalLearner):
                transform.fit_incremental(data)
            data = transform.transform(data)
