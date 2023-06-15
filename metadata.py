import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
from ruamel.yaml import YAML


@dataclass
class AgencMetadatum:
    name: str
    description: str
    min: float
    max: float
    quantity: str
    unit: str


@dataclass
class AgencMetadata:
    data_set_name: str
    uri: str
    columns: List[AgencMetadatum]


def construct_path(uri, data_set_name):
    # Check URI

    # if uri is path
    path = os.path.abspath(os.path.join(os.path.expanduser(uri), data_set_name))

    return path

with open("processed_data_metadata.yml") as f:
    content = YAML(typ="safe").load(f)

columns = []
for i in content["columns"]:
    columns.append(
        AgencMetadatum(
            i.get("name", "unnamed"),
            i.get("description", ""),
            i.get("type", ""),
            i.get("min", 0.0),
            i.get("max", 0.0),
            i.get("quantity", ""),
            i.get("unit", ""),
        )
    )

metadata = AgencMetadata(
    content.get("data_set_name", ""),
    content.get("URI", ""),
    columns, 
)

# TODO: Features
print(metadata.data_set_name)

data = pd.read_csv(construct_path(metadata.uri, metadata.data_set_name))

print(data)