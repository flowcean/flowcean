import logging
from pathlib import Path

import polars as pl

from flowcean.environments.dataset import Dataset

logger = logging.getLogger(__name__)


class CsvDataLoader(Dataset):
    """DataLoader for CSV files."""

    def __init__(self, path: str | Path, separator: str = ",") -> None:
        """Initialize the CsvDataLoader.

        Args:
            path: Path to the CSV file.
            separator: Value separator. Defaults to ",".
        """
        data = pl.read_csv(path, separator=separator)
        data.columns = [column_name.strip() for column_name in data.columns]
        super().__init__(data)
