import logging
from pathlib import Path
from typing import Self, override

import polars as pl

from flowcean.core import OfflineEnvironment
from flowcean.core.environment import NotLoadedError

logger = logging.getLogger(__name__)


class CsvDataLoader(OfflineEnvironment):
    """DataLoader for CSV files."""

    path: Path
    separator: str
    data: pl.DataFrame | None = None

    def __init__(self, path: str | Path, separator: str = ",") -> None:
        """Initialize the CsvDataLoader.

        Args:
            path: Path to the CSV file.
            separator: Value separator. Defaults to ",".
        """
        self.path = Path(path)
        self.separator = separator

    @override
    def load(self) -> Self:
        logger.info("Loading data from %s", self.path)
        self.data = pl.read_csv(self.path, separator=self.separator)
        self.data.columns = [
            column_name.strip() for column_name in self.data.columns
        ]
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise NotLoadedError
        return self.data
