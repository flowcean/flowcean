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
    data_path: list[Path]
    test_data_path: list[Path]
    features: list[Feature]

    def load_dataset(self) -> pl.DataFrame:
        data_frame = None
        for path in self.data_path:
            if path.suffix == ".csv":
                if data_frame is None:
                    data_frame = pl.read_csv(path)
                else:
                    data_frame = pl.concat([data_frame, pl.read_csv(path)])
            else:
                supported_file_types = [".csv"]
                raise ValueError(
                    "file type of data source has to be one of"
                    f" {supported_file_types}, but got: `{path.suffix}`"
                )
        return data_frame

    def load_test_dataset(self) -> pl.DataFrame:
        data_frame = None
        if self.test_data_path is None:
            return None
        for path in self.test_data_path:
            if path.suffix == ".csv":
                if data_frame is None:
                    data_frame = pl.read_csv(path)
                else:
                    data_frame = pl.concat([data_frame, pl.read_csv(path)])
            else:
                supported_file_types = [".csv"]
                raise ValueError(
                    "file type of data source has to be one of"
                    f" {supported_file_types}, but got: `{path.suffix}`"
                )
        return data_frame
