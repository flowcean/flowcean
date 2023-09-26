from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class Feature:
    name: str
    description: str | None = None
    kind: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    quantity: str | None = None
    unit: str | None = None


@dataclass
class Metadata:
    data_path: Path
    features: list[Feature]

    def load_dataset(self) -> pl.DataFrame:
        if self.data_path.suffix == ".csv":
            return pl.read_csv(self.data_path)
        supported_file_types = [".csv"]
        raise ValueError(
            "file type of data source has to be one of"
            f" {supported_file_types}, but got: `{self.data_path.suffix}`",
        )
