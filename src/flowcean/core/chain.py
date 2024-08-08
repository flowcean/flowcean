from __future__ import annotations

from typing import override

import polars as pl

from flowcean.core.learner import (
    UnsupervisedIncrementalLearner,
    UnsupervisedLearner,
)

from .transform import Transform


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
    def transform(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
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
        data_internal: pl.DataFrame | pl.LazyFrame = data
        for transform in self.transforms:
            if isinstance(transform, UnsupervisedLearner):
                if isinstance(data_internal, pl.LazyFrame):
                    data_internal = data_internal.collect()
                transform.fit(data_internal)
            data_internal = transform.transform(data_internal)

    @override
    def fit_incremental(self, data: pl.DataFrame) -> None:
        data_internal: pl.DataFrame | pl.LazyFrame = data
        for transform in self.transforms:
            if isinstance(transform, UnsupervisedIncrementalLearner):
                if isinstance(data_internal, pl.LazyFrame):
                    data_internal = data_internal.collect()
                transform.fit_incremental(data_internal)
            data_internal = transform.transform(data_internal)
