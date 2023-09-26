"""
This is the main entry point for the CLI. It is responsible for loading the
experiment file, loading the dataset, loading the transforms, learner and
metrics, and running the experiment.
"""

import argparse
from functools import reduce
from pathlib import Path

import polars as pl

from agenc.data.split import train_test_split
from agenc.experiment import Experiment


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

    dataset = reduce(
        lambda dataset, transform: transform(dataset),
        experiment.data.transforms,
        dataset,
    )

    train_data, test_data = train_test_split(
        dataset,
        experiment.data.train_test_split,
    )

    experiment.learner.train(
        train_data.select(experiment.data.inputs).to_numpy(),
        train_data.select(experiment.data.outputs).to_numpy(),
    )

    predictions = experiment.learner.predict(
        test_data.select(experiment.data.inputs).to_numpy()
    )

    for metric in experiment.metrics:
        result = metric(
            test_data.select(experiment.data.outputs).to_numpy(), predictions
        )
        print(f"{metric.name}: {result}")


if __name__ == "__main__":
    main()
