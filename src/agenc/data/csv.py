from pathlib import Path

import polars as pl
from typing_extensions import override

from agenc.core import DataLoader


class CsvDataLoader(DataLoader):
    """DataLoader for CSV files."""

    def __init__(self, path: str | Path):
        """Initialize the CsvDataLoader.

        Args:
            path: Path to the CSV file.
        """
        self.path = Path(path)

    @override
    def load(self) -> pl.DataFrame:
        return pl.read_csv(self.path)
