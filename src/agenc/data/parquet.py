from pathlib import Path

import polars as pl
from typing_extensions import override

from agenc.core import DataLoader


class ParquetDataLoader(DataLoader):
    """DataLoader for Parquet files."""

    def __init__(self, path: str | Path):
        """Initialize the ParquetDataLoader.

        Args:
            path: Path to the Parquet file.
        """
        self.path = Path(path)

    @override
    def load(self) -> pl.DataFrame:
        return pl.read_parquet(self.path)
