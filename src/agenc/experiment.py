from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, List, Union

from ruamel.yaml import YAML

from agenc.data.metadata import Metadata, _file_uri_to_path
from agenc.dynamic_loader import load_instance
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
    seed: int
    metadata: Metadata
    learner: Learner
    data: Data
    metrics: List[Metric]

    @classmethod
    def load_from_path(cls, path: Union[str, os.PathLike]) -> "Experiment":
        path = Path(path)
        content = YAML(typ="safe").load(path)
        seed = content["seed"]
        metadata = Metadata.load_from_path(
            _file_uri_to_path(content["data"]["metadata"], path.parent)
        )
        learner = _load_instance_from_yaml(content["learner"])
        transforms = [
            _load_instance_from_yaml(transform)
            for transform in content["data"].get("transforms", [])
        ]
        data = Data(
            inputs=content["data"]["inputs"],
            outputs=content["data"]["outputs"],
            train_test_split=content["data"]["train_test_split"],
            transforms=transforms,
        )
        metrics = [
            _load_instance_from_yaml(metric)
            for metric in content.get("metrics", [])
        ]

        return cls(
            seed=seed,
            metadata=metadata,
            learner=learner,
            data=data,
            metrics=metrics,
        )


def _load_instance_from_yaml(entry: dict | str) -> Any:
    if isinstance(entry, dict):
        return load_instance(entry["class_path"], entry.get("arguments", {}))
    elif isinstance(entry, str):
        return load_instance(entry, {})
    else:
        raise ValueError(
            f"Expected either a dictionary or a string, got {type(entry)}"
        )
