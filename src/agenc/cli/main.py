"""Command line interface for running experiments.

This is the main entry point for the CLI. It is responsible for loading the
experiment file, loading the dataset, loading the transforms, learner and
metrics, and running the experiment.
"""

import argparse
from functools import reduce
from pathlib import Path

from agenc.core import train_test_split

from .yaml import load_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Path to experiment file",
    )
    arguments = parser.parse_args()

    experiment = load_experiment(arguments.experiment)

    dataset = experiment.metadata.load_dataset()

    train_data, test_data = train_test_split(
        dataset,
        experiment.train_test_split,
    )

    train_data = reduce(
        lambda dataset, transform: transform(dataset),
        experiment.transforms,
        train_data,
    )
    test_data = reduce(
        lambda dataset, transform: transform(dataset),
        experiment.transforms,
        test_data,
    )

    experiment.learner.train(
        train_data.select(experiment.inputs).to_numpy(),
        train_data.select(experiment.outputs).to_numpy(),
    )

    predictions = experiment.learner.predict(
        test_data.select(experiment.inputs).to_numpy(),
    )

    for metric in experiment.metrics:
        result = metric(
            test_data.select(experiment.outputs).to_numpy(),
            predictions,
        )
        print(f"{metric.__class__.__name__}: {result}")


if __name__ == "__main__":
    main()
