from pathlib import Path

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # noqa: UP035

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
        super().__init__(data.lazy())

    @override
    def observe(self) -> pl.LazyFrame:
        return self.data
