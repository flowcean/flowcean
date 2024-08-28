from __future__ import annotations

from typing import TYPE_CHECKING, override

from .transform import Transform

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


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
