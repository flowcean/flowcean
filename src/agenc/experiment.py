from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Union

from ruamel.yaml import YAML

from agenc.metadata import AgencMetadata, _file_uri_to_path


@dataclass
class Learner:
    class_path: str
    init_arguments: dict


@dataclass
class Preprocessor:
    class_path: str
    init_arguments: dict


@dataclass
class Data:
    inputs: List[str]
    outputs: List[str]
    train_test_split: float
    preprocessors: List[Preprocessor]


@dataclass
class Metric:
    class_path: str
    init_arguments: dict


@dataclass
class Experiment:
    random_state: int
    metadata: AgencMetadata
    learner: Learner
    data: Data
    metrics: List[Metric]

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
            init_arguments=content["learner"].get("init_arguments", {}),
        )
        preprocessors = [
            Preprocessor(
                class_path=preprocessor["class_path"],
                init_arguments=preprocessor.get("init_arguments", {}),
            )
            for preprocessor in content["data"].get("preprocessors", [])
        ]
        data = Data(
            inputs=content["data"]["inputs"],
            outputs=content["data"]["outputs"],
            train_test_split=content["data"]["train_test_split"],
            preprocessors=preprocessors,
        )
        metrics = [
            Metric(
                class_path=metric["class_path"],
                init_arguments=metric.get("init_arguments", {}),
            )
            for metric in content.get("metrics", [])
        ]

        return cls(
            random_state=random_state,
            metadata=metadata,
            learner=learner,
            data=data,
            metrics=metrics,
        )
