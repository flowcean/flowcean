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

import polars as pl

from agenc.cli.experiment import LearnerSpecification
from agenc.core import Chain, DataLoader, Learner, Metric, Model
from agenc.data.split import TrainTestSplit
from agenc.transforms import Select

from . import logging, runtime_configuration
from .yaml import load_experiment


def parse_arguments() -> argparse.Namespace:
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
    return parser.parse_args()


def load_runtime_configuration(path: Path | None) -> None:
    if path is not None:
        runtime_configuration.load_from_file(path)
    elif exists(path := Path.cwd() / "runtime.yaml"):
        runtime_configuration.load_from_file(path)


def load_data(
    data_loader: DataLoader,
    test_data_loader: DataLoader | TrainTestSplit,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    dataset = data_loader.load()
    if isinstance(test_data_loader, TrainTestSplit):
        return test_data_loader(dataset)
    return dataset, test_data_loader.load()


def train_learner(
    learner: Learner,
    train_data: pl.DataFrame,
    specification: LearnerSpecification,
) -> Model:
    logger = _logging.getLogger(__name__)
    logger.info(f"Start training of `{specification.name}`")
    model = learner.train(
        train_data,
        train_data,
    )
    logger.info("Finished training")
    if specification.save_path is not None:
        logger.info(f"Saving model to `{specification.save_path}`")
        specification.save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(specification.save_path)
    return model


def main() -> None:
    sys.path.append(str(Path.cwd()))

    arguments = parse_arguments()
    load_runtime_configuration(arguments.configuration)

    logging.initialize(level=arguments.verbose)
    logger = _logging.getLogger(__name__)

    logger.info(f"Loading experiment from `{arguments.experiment}`")
    experiment = load_experiment(arguments.experiment)

    data_loader: DataLoader = experiment.data_loader.create()
    test_data_loader = experiment.test_data_loader.create()
    transforms = Chain(
        *[transform.create() for transform in experiment.transforms]
    )
    learners: list[Learner] = [
        learner.create() for learner in experiment.learners
    ]
    metrics: list[Metric] = [metric.create() for metric in experiment.metrics]

    logger.info("Loading data")
    train_data, test_data = load_data(data_loader, test_data_loader)

    logger.info("Transforming data")
    train_data = transforms.fit_then_transform(train_data)
    test_data = transforms(test_data)

    select_inputs = Select(experiment.inputs)
    select_outputs = Select(experiment.outputs)

    models = []
    for specification, learner in zip(
        experiment.learners,
        learners,
        strict=True,
    ):
        input_data = select_inputs(train_data)
        model = train_learner(learner, input_data, specification)
        models.append(model)

    for specification, model in zip(
        experiment.learners,
        models,
        strict=True,
    ):
        logger.info(f"Predicting with `{specification.name}`")
        input_data = select_inputs(test_data)
        predictions = model.predict(input_data)

        for metric in metrics:
            result = metric(
                select_outputs(test_data).to_numpy(),
                predictions,
            )
            print(f"{metric.__class__.__name__}: {result}")


if __name__ == "__main__":
    main()
