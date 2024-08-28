import json
from pathlib import Path
from typing import override

import polars as pl

from flowcean.environments.dataset import Dataset


class JsonDataLoader(Dataset):
    data: pl.DataFrame

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        with path.open() as file:
            json_content = json.load(file)
        self.data = pl.DataFrame(json_content)

    @override
    def observe(self) -> pl.DataFrame:
        return self.data
