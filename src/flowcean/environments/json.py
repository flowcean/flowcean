import json
from pathlib import Path
from typing import Self, override

import polars as pl

from flowcean.core import OfflineEnvironment
from flowcean.core.environment import NotLoadedError


class JsonDataLoader(OfflineEnvironment):
    path: Path
    data: pl.DataFrame | None = None

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @override
    def load(self) -> Self:
        with self.path.open() as file:
            json_content = json.load(file)

        # Check if any of the entries in the dict is *not* a list and treat the
        # whole dict as a single entry in that case
        if isinstance(json_content, dict) and any(
            not isinstance(value, list) for value in json_content.values()
        ):
            self.data = pl.DataFrame([json_content])
        else:
            self.data = pl.DataFrame(json_content)
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise NotLoadedError
        return self.data
