import argparse
from functools import reduce
from pathlib import Path

from agenc.data import Dataset
from agenc.dyna_loader import load_class
from agenc.experiment import Experiment
from agenc.transforms import Transform
from agenc.learner import Learner
from agenc.metrics import Metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Path to experiment file",
    )
    arguments = parser.parse_args()

    experiment = Experiment.load_from_path(arguments.experiment)

    dataset = Dataset.from_experiment(experiment)

    transforms: list[Transform] = [
        load_class(transform.class_path, transform.init_arguments)
        for transform in experiment.data.transforms
    ]
    learner: Learner = load_class(
        experiment.learner.class_path,
        experiment.learner.init_arguments,
    )
    metrics: list[Metric] = [
        load_class(metric.class_path, metric.init_arguments)
        for metric in experiment.metrics
    ]

    dataset = reduce(
        lambda dataset, transform: transform(dataset),
        transforms,
        dataset,
    )

    train_data, test_data = dataset.train_test_split(
        experiment.data.train_test_split,
        experiment.random_state,
    )

    learner.train(train_data)

    # TODO: the prediction should not get the ground truth
    predictions = learner.predict(test_data)

    for metric in metrics:
        print(f"{metric.name}: {metric(test_data.outputs(), predictions)}")


if __name__ == "__main__":
    main()
