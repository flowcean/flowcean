from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flowcean.core import Data, Metric


class LazyMixin:
    """If input is a polars.LazyFrame, collect() it before passing on."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def prepare(self: Metric, data: Data) -> Data:
        if isinstance(data, pl.LazyFrame):
            data = data.collect(engine="streaming")
        return super().prepare(data)


class SelectMixin:
    """Select only specified columns from DataFrame-like objects."""

    features: Sequence[str] | None = None

    def __init__(
        self,
        *,
        features: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.features = features

    def prepare(self: HasFeaturesAndPrepare, data: Data) -> Data:
        if self.features is not None:
            data = data.select(self.features)
        return super().prepare(data)


class HasFeaturesAndPrepare(Protocol):
    features: Sequence[str] | None

    def prepare(self, data: Data) -> Data: ...
