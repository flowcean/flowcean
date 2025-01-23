#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
#     "matplotlib",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../.." , editable = true }
# ///

# system libraries
import argparse
import logging
import math
import time

# third-party libraries
import matplotlib.pyplot as plt
from polars import DataFrame

# flowcean libraries
import flowcean.cli
from flowcean.environments.parquet import ParquetDataLoader
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.lightning import LightningLearner, MultilayerPerceptron
from flowcean.learners.regression_tree import RegressionTree
from flowcean.metrics.regression import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms import Filter, Flatten, Resample, Select
from flowcean.transforms.filter import (
    And,  # noqa: F401
    Not,  # noqa: F401
    Or,  # noqa: F401
)  # used for filter evaluation

# start logger
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    flowcean.cli.initialize_logging(parse_arguments=False)

    data, inputs, outputs = load_and_prepare_data(args)
    inspect_training_data(args, data)

    if not args.no_training:
        train_and_evaluate_model(args, data, inputs, outputs)


def load_and_prepare_data(args: argparse.Namespace) -> tuple:
    logger.info("Loading data...")
    time_start = time.time()
    data = (
        ParquetDataLoader("./data/" + args.training_data)
        | Select(
            [
                "p_accumulator",
                "containerWeight",
                "p_initial",
                "activeValveCount",
                "T",
            ],
        )
        | Filter(eval(args.filter))  # noqa: S307
        | Resample(args.sample_rate)
        | Flatten()
    )
    time_after_load = time.time()
    logger.info("Took %.5f s to load data", time_after_load - time_start)

    inputs = [
        "^p_accumulator_.*$",
        "activeValveCount",
        "p_initial",
        "T",
    ]
    outputs = ["containerWeight"]

    return data, inputs, outputs


def inspect_training_data(
    args: argparse.Namespace,
    data: ParquetDataLoader,
) -> None:
    if (
        args.print_overview
        or args.check_redundancy
        or args.print_data
        or args.print_row
        or args.plot_data
        or args.plot_row
    ):
        logger.info("Observing data...")
        time_start = time.time()
        observed_data = data.observe().collect()
        time_after_observe = time.time()
        logger.info(
            "Took %.5f s to observe data",
            time_after_observe - time_start,
        )

        if args.print_overview:
            print_data_overview(observed_data)

        if args.check_redundancy:
            check_data_redundancy(observed_data)

        if args.print_data:
            print_data_rows(args, observed_data)

        if args.print_row:
            print_data_row_interactively(observed_data)

        if args.plot_data:
            plot_data_rows(args, observed_data)

        if args.plot_row:
            plot_data_row_interactively(observed_data)


def print_data_overview(observed_data: DataFrame) -> None:
    logger.info("Data overview:")
    print(observed_data)


def check_data_redundancy(observed_data: DataFrame) -> None:
    logger.info("Checking for duplicated and unique output-values:")
    rows = {}
    duplicates = {}
    uniques = {}
    for row, i in zip(
        observed_data.select("containerWeight").iter_rows(),
        range(observed_data.select("containerWeight").shape[0]),
        strict=False,
    ):
        if row[0] in rows:
            if row[0] in duplicates:
                duplicates[row[0]].append(i)
            else:
                duplicates[row[0]] = [rows[row[0]], i]
            if row[0] in uniques:
                uniques.pop(row[0])
        else:
            rows[row[0]] = i
            uniques[row[0]] = i

    if duplicates:
        print(f"Duplicates ({len(duplicates)}):")
        print(duplicates)
        print(f"Uniques ({len(uniques)}):")
        print(uniques)
    else:
        print("No duplicates found.")


def print_data_rows(
    args: argparse.Namespace,
    observed_data: DataFrame,
) -> None:
    logger.info("Printing %d rows:", args.prints)

    dimension = observed_data.shape[0]
    prints = min(dimension, args.prints)

    for i, c in zip(
        range(0, dimension, int(dimension / prints)),
        range(prints),
        strict=False,
    ):
        index = i if args.print_distributed else c
        print(f"Index: {index}")
        print(
            f"Weight: {round(observed_data.select('containerWeight')
                             .row(index)[0], 3)}",
        )
        print(
            f"Active Valves: {observed_data.select('activeValveCount')
                              .row(index)[0]}",
        )
        print(
            f"Initial Pressure: {round(observed_data.select('p_initial')
                                       .row(index)[0], 3)}",
        )
        print(
            f"Temperature: {round(observed_data.select('T')
                                  .row(index)[0], 3)}",
        )
        print(
            f"Accumulated Pressures: \n{observed_data
                .select("^p_accumulator_.*$").row(index)}",
        )


def print_data_row_interactively(observed_data: DataFrame) -> None:
    logger.info("Printing rows interactively:")
    while True:
        index = input("Enter the row index to print or 'x' to quit: ")
        if index == "x":
            break
        print(
            f"Weight: {round(observed_data.select('containerWeight')
                             .row(int(index))[0], 3)}",
        )
        print(
            f"Active Valves: {observed_data.select('activeValveCount')
                              .row(int(index))[0]}",
        )
        print(
            f"Initial Pressure: {round(observed_data.select('p_initial')
                                       .row(int(index))[0], 3)}",
        )
        print(
            f"Temperature: {round(observed_data.select('T')
                                  .row(int(index))[0], 3)}",
        )
        print(
            f"Accumulated Pressures: \n{observed_data
                .select("^p_accumulator_.*$").row(int(index))}",
        )


def plot_data_rows(args: argparse.Namespace, observed_data: DataFrame) -> None:
    logger.info("Plotting %d rows:", args.plots)

    dimension = observed_data.shape[0]
    plots = min(dimension, args.plots)
    plot_rows = math.ceil(math.sqrt(plots))
    plot_cols = math.ceil(plots / plot_rows)

    plt.figure()
    for i, c in zip(
        range(0, dimension, int(dimension / plots)),
        range(plots),
        strict=False,
    ):
        plt.subplot(plot_rows, plot_cols, c + 1)
        index = i if args.plot_distributed else c
        weight = round(
            observed_data.select("containerWeight").row(index)[0],
            3,
        )
        plt.title(
            f"Weight: {weight}, Index: {index}",
        )
        plt.plot(observed_data.select("^p_accumulator_.*$").row(index))

    plt.subplots_adjust(
        hspace=0.5,
        wspace=0.5,
        left=0.1,
        right=0.95,
        top=0.9,
        bottom=0.1,
    )
    plt.show()


def plot_data_row_interactively(observed_data: DataFrame) -> None:
    logger.info("Plotting rows interactively:")

    plt.figure()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

    while True:
        index = input("Enter the row index to plot or 'x' to quit: ")
        weight = round(
            observed_data.select("containerWeight").row(int(index))[0],
            3,
        )
        active_valves = observed_data.select("activeValveCount").row(
            int(index),
        )[0]
        temperature = round(observed_data.select("T").row(int(index))[0], 3)
        if index == "x":
            break
        plt.title(
            f"Weight: {weight}, Active Valves: {active_valves}, "
            f"Temperature: {temperature}",
        )
        plt.plot(
            observed_data.select("^p_accumulator_.*$").row(int(index)),
        )
        plt.show()


def train_and_evaluate_model(
    args: argparse.Namespace,
    data: ParquetDataLoader,
    inputs: list,
    outputs: list,
) -> None:
    logger.info("Training the model:")
    time_start = time.time()

    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
        data,
    )
    if args.lightning_learner:
        learner = LightningLearner(
            MultilayerPerceptron(
                learning_rate=args.learning_rate,
                input_size=len(inputs),
                output_size=len(outputs),
            ),
        )
    else:
        learner = RegressionTree()

    model = learn_offline(
        train_env,
        learner,
        inputs,
        outputs,
    )

    report = evaluate_offline(
        model,
        test_env,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    time_after_learning = time.time()
    logger.info(
        "Took %.5f s to learn model",
        time_after_learning - time_start,
    )

    print(report)


# parse arguments and run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run the Automatic Lashing Platform example.",
    )

    # parameter

    parameter_group = parser.add_argument_group(
        "Parameter",
        "Parameter options for the training-data.",
    )
    parameter_group.add_argument(
        "--training_data",
        type=str,
        default="alp_sim_data.parquet",
        metavar="FILE",
        help="Set the training-data file. (default: alp_sim_data.parquet)",
    )
    parameter_group.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        metavar="RATE",
        help="Set the sample rate for the data. (default: 1.0) -> 15 Values",
    )
    parameter_group.add_argument(
        "--filter",
        type=str,
        default='"1==1"',
        metavar="CONDITION",
        help=(
            """Filter the data with a condition like """
            """\'And(["activeValveCount > 0", "activeValveCount < 3"])\' """
            """or \'"activeValveCount > 0"\'."""
        ),
    )

    # training-data inspection

    data_inspection_group = parser.add_argument_group(
        "Training-Data",
        "Tools to inspect the training-data.",
    )
    data_inspection_group.add_argument(
        "--print_overview",
        action="store_true",
        help="Print a short overview ot the training-data.",
    )
    data_inspection_group.add_argument(
        "--check_redundancy",
        action="store_true",
        help=(
            "Checking for duplicated and unique output-values in the "
            "training-data. (only for containerWeight)"
        ),
    )
    data_inspection_group.add_argument(
        "--print_data",
        action="store_true",
        help="Print a number of rows of the training-data.",
    )
    data_inspection_group.add_argument(
        "--prints",
        type=int,
        default=10,
        metavar="NUMBER",
        help="Number of rows to print. (default: 10)",
    )
    data_inspection_group.add_argument(
        "--print_distributed",
        action="store_true",
        help="Print from distributed training-data.",
    )
    data_inspection_group.add_argument(
        "--print_row",
        action="store_true",
        help="Print a row of the training-data interactively.",
    )
    data_inspection_group.add_argument(
        "--plot_data",
        action="store_true",
        help="Plot the training-data.",
    )
    data_inspection_group.add_argument(
        "--plots",
        type=int,
        default=20,
        metavar="NUMBER",
        help="Number of plots to show. (default: 20)",
    )
    data_inspection_group.add_argument(
        "--plot_distributed",
        action="store_true",
        help="Plot from distributed training-data.",
    )
    data_inspection_group.add_argument(
        "--plot_row",
        action="store_true",
        help="Plot a row of the training-data interactively.",
    )

    # training

    training_group = parser.add_argument_group(
        "Model-Training",
        "Options to train the model.",
    )
    training_group.add_argument(
        "--no_training",
        action="store_true",
        help="Apply no training.",
    )
    training_group.add_argument(
        "--lightning_learner",
        action="store_true",
        help=(
            "Use the Lightning Learner with Multilayer-Perceptron "
            "instead of Regression-Tree."
        ),
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        metavar="RATE",
        help="Set the learning rate for the lightning-model. (default: 0.1)",
    )

    args = parser.parse_args()

    main(args)
