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
    data: pl.LazyFrame | None = None

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
        self.data = pl.scan_csv(self.path, separator=self.separator)
        self.data = self.data.rename(
            {
                column_name: column_name.strip()
                for column_name in self.data.collect_schema().names()
            }
        )
        return self

    @override
    def get_data(self) -> pl.DataFrame | pl.LazyFrame:
        if self.data is None:
            raise NotLoadedError
        return self.data
