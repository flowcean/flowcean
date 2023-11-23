from pathlib import Path

from ruamel.yaml import YAML

from .experiment import Experiment, InstanceSpecification, LearnerSpecification


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path)
    content = YAML(typ="safe").load(path)

    seed = content["seed"]
    data_loader = InstanceSpecification.from_dict_or_string(
        content["data"]["loader"]
    )
    test_data_loader = InstanceSpecification.from_dict_or_string(
        content["data"]["test_loader"]
    )
    inputs = content["data"]["inputs"]
    outputs = content["data"]["outputs"]
    learners = [
        LearnerSpecification.from_dict(learner)
        for learner in content["learners"]
    ]
    transforms = [
        InstanceSpecification.from_dict_or_string(transform)
        for transform in content["data"].get("transforms", [])
    ]
    metrics = [
        InstanceSpecification.from_dict_or_string(metric)
        for metric in content.get("metrics", [])
    ]

    return Experiment(
        seed=seed,
        data_loader=data_loader,
        test_data_loader=test_data_loader,
        learners=learners,
        transforms=transforms,
        inputs=inputs,
        outputs=outputs,
        metrics=metrics,
    )
