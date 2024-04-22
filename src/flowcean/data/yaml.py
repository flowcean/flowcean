from pathlib import Path
from typing import Self, override

import polars as pl
from ruamel.yaml import YAML

from flowcean.core import OfflineEnvironment
from flowcean.core.environment import NotLoadedError


class YamlDataLoader(OfflineEnvironment):
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
