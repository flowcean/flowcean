from pathlib import Path

import polars as pl
from ruamel.yaml import YAML
from typing_extensions import Self, override

from agenc.core import OfflineDataLoader
from agenc.core.environment import NotLoadedError


class YamlDataLoader(OfflineDataLoader):
    path: Path
    data: pl.DataFrame | None = None

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @override
    def load(self) -> Self:
        self.data = pl.DataFrame(YAML(typ="safe").load(self.path))
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise NotLoadedError
        return self.data
