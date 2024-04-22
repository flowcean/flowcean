from pathlib import Path
from typing import Self, override

import polars as pl

from flowcean.core import OfflineEnvironment
from flowcean.core.environment import NotLoadedError


class ParquetDataLoader(OfflineEnvironment):
    """DataLoader for Parquet files."""

    path: Path
    data: pl.DataFrame | None = None

    def __init__(self, path: str | Path) -> None:
        """Initialize the ParquetDataLoader.

        Args:
            path: Path to the Parquet file.
        """
        self.path = Path(path)

    @override
    def load(self) -> Self:
        self.data = pl.read_parquet(self.path)
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise NotLoadedError
        return self.data
