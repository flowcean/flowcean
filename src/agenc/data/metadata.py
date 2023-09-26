from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import polars as pl
from ruamel.yaml import YAML


@dataclass
class Feature:
    name: str
    description: Optional[str] = None
    kind: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    quantity: Optional[str] = None
    unit: Optional[str] = None


@dataclass
class Metadata:
    data_path: Path
    features: List[Feature]

    @classmethod
    def load_from_path(cls, path: Union[str, os.PathLike]) -> "Metadata":
        path = Path(path)
        content = YAML(typ="safe").load(path)

        path = _file_uri_to_path(content["uri"], path.parent)
        features = [
            Feature(
                name=feature["name"],
                description=feature.get("description"),
                kind=feature.get("kind"),
                min=feature.get("min"),
                max=feature.get("max"),
                quantity=feature.get("quantity"),
                unit=feature.get("unit"),
            )
            for feature in content["features"]
        ]

        return cls(data_path=path, features=features)

    def load_dataset(self) -> pl.DataFrame:
        if self.data_path.suffix == ".csv":
            data_frame = pl.read_csv(self.data_path)
            return data_frame
        else:
            supported_file_types = [".csv"]
            raise ValueError(
                "file type of data source has to be one of"
                f" {supported_file_types}, but got: `{self.data_path.suffix}`"
            )


def _file_uri_to_path(uri: str, root: Path) -> Path:
    url = urlparse(uri)
    if url.scheme != "file":
        raise ValueError(
            "only local files are supported as data source, but got:"
            f" `{url.scheme}`"
        )
    data_source = Path(url.path)
    if not data_source.is_absolute():
        data_source = (root / data_source).absolute()
    return data_source
