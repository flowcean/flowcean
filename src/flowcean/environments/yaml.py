from pathlib import Path

import polars as pl
from ruamel.yaml import YAML

from flowcean.environments.dataset import Dataset


class YamlDataLoader(Dataset):
    """DataLoader for YAML files."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the YamlDataLoader.

        Args:
            path: Path to the YAML file.
        """
        data = pl.DataFrame(YAML(typ="safe").load(path))
        super().__init__(data)
