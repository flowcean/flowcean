from pathlib import Path

import polars as pl
from ruamel.yaml import YAML
from typing_extensions import override

from agenc.core import DataLoader


class YamlDataLoader(DataLoader):
    """DataLoader for yaml files."""

    def __init__(self, path: str | Path):
        """Initialize the YamlDataLoader.

        Args:
            path: Path to the Yaml file.
        """
        self.path = Path(path)

    @override
    def load(self) -> pl.DataFrame:
        return pl.DataFrame(YAML(typ="safe").load(self.path))
