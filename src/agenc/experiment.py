from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Union

from ruamel.yaml import YAML

from agenc.data.metadata import Metadata, _file_uri_to_path
from agenc.dynamic_loader import load_class
from agenc.learner import Learner
from agenc.metrics import Metric
from agenc.transforms import Transform


@dataclass
class Data:
    inputs: List[str]
    outputs: List[str]
    train_test_split: float
    transforms: List[Transform]


@dataclass
class Experiment:
    random_state: int
    metadata: Metadata
    learner: Learner
    data: Data
    metrics: List[Metric]

    @classmethod
    def load_from_path(cls, path: Union[str, os.PathLike]) -> "Experiment":
        path = Path(path)
        content = YAML(typ="safe").load(path)
        random_state = content["random_state"]
        metadata = Metadata.load_from_path(
            _file_uri_to_path(content["data"]["metadata"], path.parent)
        )
        learner = load_class(
            content["learner"]["class_path"],
            content["learner"].get("init_arguments", {}),
        )
        transforms = [
            load_class(
                transform["class_path"], transform.get("init_arguments", {})
            )
            for transform in content["data"].get("transforms", [])
        ]
        data = Data(
            inputs=content["data"]["inputs"],
            outputs=content["data"]["outputs"],
            train_test_split=content["data"]["train_test_split"],
            transforms=transforms,
        )
        metrics = [
            load_class(metric["class_path"], metric.get("init_arguments", {}))
            for metric in content.get("metrics", [])
        ]

        return cls(
            random_state=random_state,
            metadata=metadata,
            learner=learner,
            data=data,
            metrics=metrics,
        )
