from pathlib import Path

import polars as pl
from typing_extensions import override

from agenc.core import DataLoader


class CsvDataLoader(DataLoader):
    """DataLoader for CSV files."""

    def __init__(self, path: str | Path, seperator: str = ","):
        """Initialize the CsvDataLoader.

        Args:
            path: Path to the CSV file.
            seperator: Value seperator. Defaults to ",".
        """
        self.path = Path(path)
        self.seperator = seperator

    @override
    def load(self) -> pl.DataFrame:
        data = pl.read_csv(self.path, separator=self.seperator)
        data.columns = [column_name.strip() for column_name in data.columns]
        return data
