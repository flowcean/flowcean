from pathlib import Path
from typing import override

import polars as pl

from flowcean.environments.dataset import Dataset


class JsonDataLoader(Dataset):
    """DataLoader for JSON files."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the JsonDataLoader.

        Args:
            path: Path to the JSON file.
        """
        data = pl.read_json(path)
        super().__init__(data)

    @override
    def observe(self) -> pl.DataFrame:
        return self.data
