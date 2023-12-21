from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from agenc.core import Feature, Metadata

from ._uri import file_uri_to_path
from .experiment import Experiment, InstanceSpecification, LearnerSpecification


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path)
    content = YAML(typ="safe").load(path)
    return _load(content, path.parent)


def _load(content: Any, working_directory: Path) -> Experiment:
    seed = content["seed"]
    metadata = load_metadata(
        file_uri_to_path(content["data"]["metadata"], working_directory)
    )
    learner = LearnerSpecification.from_dict_or_string(content["learner"])
    transforms = [
        InstanceSpecification.from_dict_or_string(transform)
        for transform in content["data"].get("transforms", [])
    ]
    inputs = content["data"]["inputs"]
    outputs = content["data"]["outputs"]
    train_test_split = content["data"]["train_test_split"]
    metrics = [
        InstanceSpecification.from_dict_or_string(metric)
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
