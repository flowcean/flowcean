from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Union

from ruamel.yaml import YAML

from agenc.metadata import AgencMetadata, _file_uri_to_path


@dataclass
class Learner:
    class_path: str
    parameters: dict


@dataclass
class Data:
    inputs: List[str]
    outputs: List[str]
    train_test_split: float


@dataclass
class Experiment:
    random_state: int
    metadata: AgencMetadata
    learner: Learner
    data: Data

    @classmethod
    def load_from_path(cls, path: Union[str, os.PathLike]) -> "Experiment":
        path = Path(path)
        content = YAML(typ="safe").load(path)
        random_state = content["random_state"]
        metadata = AgencMetadata.load_from_path(
            _file_uri_to_path(content["data"]["metadata"], path.parent)
        )
        learner = Learner(
            class_path=content["learner"]["class_path"],
            parameters=content["learner"]["parameters"],
        )
        data = Data(
            inputs=content["data"]["inputs"],
            outputs=content["data"]["outputs"],
            train_test_split=content["data"]["train_test_split"],
        )

        return cls(
            random_state=random_state,
            metadata=metadata,
            learner=learner,
            data=data,
        )
