import logging
from pathlib import Path

import polars as pl

from .dataset import Dataset

logger = logging.getLogger(__name__)


class CsvDataLoader(Dataset):
    """DataLoader for CSV files."""

    def __init__(self, path: str | Path, separator: str = ",") -> None:
        """Initialize the CsvDataLoader.

        Args:
            path: Path to the CSV file.
            separator: Value separator. Defaults to ",".
        """
        data = pl.scan_csv(path, separator=separator)
        data = data.rename(lambda column_name: column_name.strip())
        super().__init__(data)
