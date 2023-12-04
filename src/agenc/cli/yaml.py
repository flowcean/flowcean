from pathlib import Path
from typing import Any, TypedDict

from ruamel.yaml import YAML

from agenc.core import Experiment, Feature, Metadata

from ._dynamic_loader import load_instance
from ._uri import file_uri_to_path


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path)
    content = YAML(typ="safe").load(path)
    seed = content["seed"]
    metadata = load_metadata(
        file_uri_to_path(content["data"]["metadata"], path.parent)
    )
    learner = _load_instance_from_yaml(content["learner"])
    transforms = [
        _load_instance_from_yaml(transform)
        for transform in content["data"].get("transforms", [])
    ]
    inputs = content["data"]["inputs"]
    outputs = content["data"]["outputs"]
    train_test_split = content["data"]["train_test_split"]
    metrics = [
        _load_instance_from_yaml(metric)
        for metric in content.get("metrics", [])
    ]

    return Experiment(
        seed=seed,
        metadata=metadata,
        learner=learner,
        transforms=transforms,
        inputs=inputs,
        outputs=outputs,
        train_test_split=train_test_split,
        metrics=metrics,
    )


def load_metadata(path: str | Path) -> Metadata:
    path = Path(path)
    content = YAML(typ="safe").load(path)

    uri = content.get("uri")
    if isinstance(uri, str):
        uri = [uri]
    paths = [file_uri_to_path(uri, path.parent) for uri in uri]
    test_paths = [
        file_uri_to_path(uri, path.parent)
        for uri in content.get("test_uri", [])
    ]

    features = [
        Feature(
            name=feature["name"],
            description=feature.get("description"),
            kind=feature.get("kind"),
            minimum=feature.get("min"),
            maximum=feature.get("max"),
            quantity=feature.get("quantity"),
            unit=feature.get("unit"),
        )
        for feature in content["features"]
    ]

    return Metadata(
        data_path=paths, test_data_path=test_paths, features=features
    )


class InstanceConfiguration(TypedDict):
    class_path: str
    arguments: dict[str, Any]


def _load_instance_from_yaml(entry: InstanceConfiguration | str) -> Any:
    if isinstance(entry, dict):
        return load_instance(entry["class_path"], entry.get("arguments", {}))
    if isinstance(entry, str):
        return load_instance(entry, {})
    raise ValueError(
        f"Expected either a dictionary or a string, got {type(entry)}"
    )
