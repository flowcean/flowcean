"""Command line interface for running experiments.

This is the main entry point for the CLI. It is responsible for loading the
experiment file, loading the dataset, loading the transforms, learner and
metrics, and running the experiment.
"""

import argparse
import logging as _logging
import sys
from os.path import exists
from pathlib import Path

from agenc.core import Chain, DataLoader, Learner, Metric
from agenc.data.split import TrainTestSplit

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
    cwd = Path.cwd()
    sys.path.append(str(cwd))

    if arguments.configuration is not None:
        runtime_configuration.load_from_file(arguments.configuration)
    elif exists(path := cwd / "runtime.yaml"):
        runtime_configuration.load_from_file(path)

    logging.inititialize(level=arguments.verbose)
    logger = _logging.getLogger(__name__)

    logger.info(f"Loading experiment from `{arguments.experiment}`")
    experiment = load_experiment(arguments.experiment)

    logger.info(f"Loading data using `{experiment.data_loader.class_path}`")
    data_loader: DataLoader = experiment.data_loader.load()
    dataset = data_loader.load()

    test_data_loader = experiment.test_data_loader.load()
    if isinstance(test_data_loader, DataLoader):
        logger.info("Loading test data using data loader")
        train_data = dataset
        test_data = test_data_loader.load()
    elif isinstance(test_data_loader, TrainTestSplit):
        logger.info("Splitting data into train and test set")
        train_data, test_data = test_data_loader(dataset)
    else:
        raise ValueError(
            "test_data_loader has to be of type DataLoader or"
            f" TrainTestSplit, but got: `{test_data_loader}`"
        )

    transforms = Chain(
        *[transform.load() for transform in experiment.transforms]
    )
    learners: list[Learner] = [
        learner.load() for learner in experiment.learners
    ]
    metrics: list[Metric] = [metric.load() for metric in experiment.metrics]

    logger.info("Fitting transforms")
    transforms.fit(train_data)

    logger.info("Applying transforms")
    train_data = transforms(train_data)
    test_data = transforms(test_data)

    for specification, learner in zip(
        experiment.learners,
        learners,
        strict=True,
    ):
        if specification.load_path is not None:
            logger.info(f"Loading learner from `{specification.load_path}`")
            learner.load(specification.load_path)
        if specification.train:
            logger.info(f"Start training of `{specification.name}`")
            learner.train(
                train_data.select(experiment.inputs),
                train_data.select(experiment.outputs),
            )
            if specification.save_path is not None:
                logger.info(f"Saving learner to `{specification.save_path}`")
                specification.save_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
            logger.info("Finished training")
            if specification.save_path is not None:
                logger.info(f"Saving learner to `{specification.save_path}`")
                specification.save_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                learner.save(specification.save_path)

    for specification, learner in zip(
        experiment.learners,
        learners,
        strict=True,
    ):
        logger.info(f"Predicting with `{specification.name}`")
        predictions = learner.predict(
            test_data.select(experiment.inputs),
        )

        for metric in metrics:
            result = metric(
                test_data.select(experiment.outputs).to_numpy(),
                predictions,
            )
            print(f"{metric.__class__.__name__}: {result}")


if __name__ == "__main__":
    main()
