import json
from pathlib import Path

import polars as pl
from typing_extensions import override

from agenc.core import DataLoader


class JsonDataLoader(DataLoader):
    """DataLoader for json files."""

    def __init__(self, path: str | Path):
        """Initialize the JsonDataLoader.

        Args:
            path: Path to the Json file.
        """
        self.path = Path(path)

    @override
    def load(self) -> pl.DataFrame:
        with open(self.path) as file:
            json_content = json.load(file)
        return pl.DataFrame(json_content)
