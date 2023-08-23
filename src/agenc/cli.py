import argparse
from functools import reduce
from pathlib import Path

import polars as pl

from agenc.data.split import train_test_split
from agenc.dynamic_loader import load_class
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

    dataset: pl.DataFrame = experiment.metadata.load_dataset()

    test_data: pl.DataFrame = experiment.metadata.load_test_dataset()

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

    if test_data is None:
        train_data, test_data = train_test_split(
            dataset,
            experiment.data.train_test_split,
        )
    else:
        train_data = dataset

    learner.train(
        train_data.select(experiment.data.inputs).to_numpy(),
        train_data.select(experiment.data.outputs).to_numpy(),
    )

    predictions = learner.predict(
        test_data.select(experiment.data.inputs).to_numpy()
    )

    for metric in metrics:
        result = metric(
            test_data.select(experiment.data.outputs).to_numpy(), predictions
        )
        print(f"{metric.name}: {result}")


if __name__ == "__main__":
    main()
