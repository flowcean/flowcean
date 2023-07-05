from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Union, Optional
from urllib.parse import urlparse

from ruamel.yaml import YAML


@dataclass
class AgencMetadatum:
    name: str
    description: Optional[str]
    kind: Optional[str]
    min: Optional[float]
    max: Optional[float]
    quantity: Optional[str]
    unit: Optional[str]


@dataclass
class AgencMetadata:
    data_path: Path
    columns: List[AgencMetadatum]

    @classmethod
    def load_from_path(cls, path: Union[str, os.PathLike]) -> "AgencMetadata":
        path = Path(path)
        content = YAML(typ="safe").load(path)

        path = _file_uri_to_path(content["uri"], path.parent)
        columns = [
            AgencMetadatum(
                name=column["name"],
                description=column.get("description"),
                kind=column.get("kind"),
                min=column.get("min"),
                max=column.get("max"),
                quantity=column.get("quantity"),
                unit=column.get("unit"),
            )
            for column in content["columns"]
        ]

        return cls(
            data_path=path,
            columns=columns,
        )


def _file_uri_to_path(uri: str, root: Path) -> Path:
    url = urlparse(uri)
    if url.scheme != "file":
        raise ValueError(
            f"only local files are supported as data source, but got: `{url.scheme}`"
        )
    data_source = Path(url.path)
    if not data_source.is_absolute():
        data_source = (root / data_source).absolute()
    return data_source
