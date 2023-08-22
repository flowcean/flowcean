from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import polars as pl
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
class AgencFeature:
    metadatum: AgencMetadatum
    import_str: str
    params: List[str]


@dataclass
class AgencMetadata:
    data_path: Path
    columns: List[AgencMetadatum]
    features: List[AgencFeature]

    @classmethod
    def load_from_path(cls, path: Union[str, os.PathLike]) -> "AgencMetadata":
        path = Path(path)
        content = YAML(typ="safe").load(path)

        paths = []
        for i in range(len(content["uri"])):
            paths.append(_file_uri_to_path(content["uri"][i], path.parent))
               
        print(path)
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
        features = [
            AgencFeature(
                AgencMetadatum(
                    name=feature["name"],
                    description=feature.get("description"),
                    kind=feature.get("kind"),
                    min=feature.get("min"),
                    max=feature.get("max"),
                    quantity=feature.get("quantity"),
                    unit=feature.get("unit"),
                ),
                import_str=feature["import_str"],
                params=[attr for attr in feature.get("params", [])],
            )
            for feature in content.get("features", [])
        ]

        return cls(data_path=paths, columns=columns, features=features)


    def load_csvs(self) -> List[pl.DataFrame]:
        data_frames = []
        for path in self.data_path:
            data_frames.append(pl.read_csv(path))
        return data_frames
    

    # def concatenate_dataframes(self) -> pl.DataFrame:
    #     data_frames = self.load_csvs()
    #     return data_frames[0].vcat(data_frames[1])
    

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
