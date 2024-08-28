from __future__ import annotations

from typing import TYPE_CHECKING, override

from flowcean.core.environment.offline import OfflineEnvironment

if TYPE_CHECKING:
    import polars as pl


class Dataset(OfflineEnvironment):
    data: pl.DataFrame

    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data

    @override
    def observe(self) -> pl.DataFrame:
        return self.data

    def __len__(self) -> int:
        return len(self.data)
