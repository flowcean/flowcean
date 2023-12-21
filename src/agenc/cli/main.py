"""Command line interface for running experiments.

This is the main entry point for the CLI. It is responsible for loading the
experiment file, loading the dataset, loading the transforms, learner and
metrics, and running the experiment.
"""

import argparse
import logging as _logging
from functools import reduce
from os.path import exists
from pathlib import Path

from agenc.core import Learner, Metric, Transform, train_test_split

from . import logging, runtime_configuration
from .yaml import load_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Path to experiment file",
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        required=False,
        help="Path to runtime configuration",
    )
    parser.add_argument(
        "--verbose",
        action="store_const",
        const=_logging.DEBUG,
        help="increase verbosity",
    )
    arguments = parser.parse_args()

    if arguments.configuration is not None:
        runtime_configuration.load_from_file(arguments.configuration)
    elif exists(path := Path.cwd() / "runtime.yaml"):
        runtime_configuration.load_from_file(path)

    logging.inititialize(level=arguments.verbose)
    logger = _logging.getLogger(__name__)

    experiment = load_experiment(arguments.experiment)

    dataset = experiment.metadata.load_dataset()

    train_data, test_data = train_test_split(
        dataset,
        experiment.train_test_split,
    )

    transforms: list[Transform] = [
        transform.load() for transform in experiment.transforms
    ]
    learner: Learner = experiment.learner.load()
    metrics: list[Metric] = [metric.load() for metric in experiment.metrics]

    train_data = reduce(
        lambda dataset, transform: transform(dataset),
        transforms,
        train_data,
    )
    test_data = reduce(
        lambda dataset, transform: transform(dataset),
        transforms,
        test_data,
    )

    if experiment.learner.load_path is not None:
        logger.info(f"Loading learner from `{experiment.learner.load_path}`")
        learner.load(experiment.learner.load_path)
    if experiment.learner.train:
        logger.info("Start training")
        learner.train(
            train_data.select(experiment.inputs).to_numpy(),
            train_data.select(experiment.outputs).to_numpy(),
        )
        if experiment.learner.save_path is not None:
            logger.info(f"Saving learner to `{experiment.learner.save_path}`")
            experiment.learner.save_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            learner.save(experiment.learner.save_path)

    predictions = learner.predict(
        test_data.select(experiment.inputs).to_numpy(),
    )

    for metric in metrics:
        result = metric(
            test_data.select(experiment.outputs).to_numpy(),
            predictions,
        )
        print(f"{metric.__class__.__name__}: {result}")


if __name__ == "__main__":
    main()
