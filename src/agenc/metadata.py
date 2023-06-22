import os
from dataclasses import dataclass
from typing import List

from ruamel.yaml import YAML


@dataclass
class AgencMetadatum:
    name: str
    description: str
    type: str
    min: float
    max: float
    quantity: str
    unit: str


@dataclass
class AgencMetadata:
    train_data_set_name: str
    test_data_set_name: str
    uri: str
    columns: List[AgencMetadatum]

    @property
    def train_data_set_path(self):
        return construct_path(self.uri, self.train_data_set_name)
    
    @property
    def test_data_set_path(self):
        return construct_path(self.uri, self.test_data_set_name)
    

def construct_path(uri, data_set_name):
    # Check URI

    # if uri is path
    path = os.path.abspath(os.path.join(os.path.expanduser(uri), data_set_name))

    return path


def read_metadata(path):
    with open(path) as f:
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
        content.get("train_data_set_name", ""),
        content.get("test_data_set_name", ""),
        content.get("URI", ""),
        columns, 
    )
    return metadata
