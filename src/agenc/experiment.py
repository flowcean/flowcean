from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Union

from ruamel.yaml import YAML

from agenc.metadata import AgencMetadata, _file_uri_to_path


@dataclass
class Learner:
    method: str
    parameters: dict


@dataclass
class Experiment:
    metadata: AgencMetadata
    inputs: List[str]
    outputs: List[str]
    learner: Learner

    @classmethod
    def load_from_path(cls, path: Union[str, os.PathLike]) -> "Experiment":
        path = Path(path)
        content = YAML(typ="safe").load(path)
        metadata = AgencMetadata.load_from_path(
            _file_uri_to_path(content["data"]["metadata"], path.parent)
        )
        learner = Learner(
            method=content["learner"]["method"],
            parameters=content["learner"]["parameters"],
        )

        return cls(
            metadata=metadata,
            inputs=content["data"]["inputs"],
            outputs=content["data"]["outputs"],
            learner=learner,
        )
