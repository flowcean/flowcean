from pathlib import Path

import polars as pl
from ruamel.yaml import YAML

from flowcean.environments.dataset import Dataset


class YamlDataLoader(Dataset):
    def __init__(self, path: str | Path) -> None:
        data = pl.DataFrame(YAML(typ="safe").load(path))
        super().__init__(data)
