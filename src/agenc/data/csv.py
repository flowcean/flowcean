import logging
from pathlib import Path

import polars as pl
from typing_extensions import Self, override

from agenc.core import OfflineDataLoader
from agenc.core.environment import NotLoadedError

logger = logging.getLogger(__name__)


class CsvDataLoader(OfflineDataLoader):
    """DataLoader for CSV files."""

    path: Path
    seperator: str
    data: pl.DataFrame | None = None

    def __init__(self, path: str | Path, seperator: str = ",") -> None:
        """Initialize the CsvDataLoader.

        Args:
            path: Path to the CSV file.
            seperator: Value seperator. Defaults to ",".
        """
        self.path = Path(path)
        self.seperator = seperator

    @override
    def load(self) -> Self:
        logger.info("Loading data from %s", self.path)
        self.data = pl.read_csv(self.path, separator=self.seperator)
        self.data.columns = [
            column_name.strip() for column_name in self.data.columns
        ]
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise NotLoadedError
        return self.data
