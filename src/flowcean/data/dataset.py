from typing import Any, Self, override

import polars as pl

from flowcean.core import OfflineEnvironment


class Dataset(OfflineEnvironment):
    data: pl.DataFrame

    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data

    @override
    def load(self) -> Self:
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int) -> tuple[Any, ...]:
        return self.data.row(key)
