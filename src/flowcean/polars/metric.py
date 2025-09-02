from collections.abc import Sequence
from typing import Any, cast

import polars as pl

from flowcean.core import Data
from flowcean.core.metric import SupportsPrepare


class LazyMixin:
    """If input is a polars.LazyFrame, collect() it before passing on."""

    def prepare(self: SupportsPrepare, data: Data) -> Data:
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

    def prepare(self, data: Data) -> Data:
        parent = cast("SupportsPrepare", super())
        if self.features is not None:
            data = data.select(list(self.features))
        return parent.prepare(data)
