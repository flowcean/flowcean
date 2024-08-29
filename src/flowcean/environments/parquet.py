from pathlib import Path

import polars as pl

from flowcean.environments.dataset import Dataset


class ParquetDataLoader(Dataset):
    """DataLoader for Parquet files."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the ParquetDataLoader.

        Args:
            path: Path to the Parquet file.
        """
        data = pl.read_parquet(path)
        super().__init__(data)
