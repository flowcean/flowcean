from typing import Any, Self, override

import polars as pl

from flowcean.core import OfflineEnvironment


class Dataset(OfflineEnvironment):
    data: pl.DataFrame | pl.LazyFrame

    def __init__(self, data: pl.DataFrame | pl.LazyFrame) -> None:
        self.data = data

    @override
    def load(self) -> Self:
        return self

    @override
    def get_data(self) -> pl.DataFrame | pl.LazyFrame:
        return self.data

    def __len__(self) -> int:
        if isinstance(self.data, pl.LazyFrame):
            self.data = self.data.collect()
        return len(self.data)

    def __getitem__(self, key: int) -> tuple[Any, ...]:
        if isinstance(self.data, pl.LazyFrame):
            self.data = self.data.collect()
        return self.data.row(key)
