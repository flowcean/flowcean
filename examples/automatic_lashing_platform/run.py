#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
#     "matplotlib",
#     "graphviz",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../.." , editable = true }
# ///

# system libraries
import argparse
import logging
import math
import os
import time
from pathlib import Path

# third-party libraries
import graphviz
import matplotlib.pyplot as plt

# flowcean libraries
import flowcean.cli
from flowcean.core import evaluate_offline, learn_offline
from flowcean.polars import (
    DataFrame,
    Derivative,
    Filter,
    Flatten,
    Resample,
    Select,
    TimeWindow,
    TrainTestSplit,
)
from flowcean.polars.transforms.filter import And, Not, Or  # noqa: F401
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanSquaredError,
    RegressionTree,
)
from flowcean.torch.lightning_learner import (
    LightningLearner,
    MultilayerPerceptron,
)

# start logger
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    flowcean.cli.initialize_logging(parse_arguments=False)

    data, inputs, outputs = load_and_prepare_data(args)
    inspect_data(args, data)

    if not args.no_training:
        train_and_evaluate_model(args, data, inputs, outputs)

    if args.show_latest_graph or args.show_graph:
        graph = get_graph(args)
        if graph:
            graph.render(view=True)


def load_and_prepare_data(args: argparse.Namespace) -> tuple:
    logger.info("Loading data...")
    time_start = time.time()
    data = (
        DataFrame.from_parquet("./data/" + args.training_data)
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
        | TimeWindow(
            time_start=args.time_window_start,
            time_end=args.time_window_end,
        )
        | Flatten()
        | Derivative("p_accumulator" if args.apply_derivative else " ")
    )
    time_after_load = time.time()
    logger.info("Took %.5f s to load data", time_after_load - time_start)

    if args.only_pressure_curve:
        inputs = ["^p_accumulator_.*$"]
    else:
        inputs = [
            "^p_accumulator_.*$",
            "activeValveCount",
            "p_initial",
            "T",
        ]

    outputs = ["containerWeight"]

    return data, inputs, outputs


def inspect_data(
    args: argparse.Namespace,
    data: DataFrame,
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
            print_overview(observed_data)

        if args.check_redundancy:
            check_redundancy(observed_data)

        if args.print_data:
            print_data(args, observed_data)

        if args.plot_data:
            plot_data(args, observed_data)

        if args.print_row:
            print_row(observed_data)

        if args.plot_row:
            plot_row(observed_data)


def print_overview(observed_data: DataFrame) -> None:
    logger.info("Data overview:")
    print(observed_data)


def check_redundancy(observed_data: DataFrame) -> None:
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


def print_data(
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
            f"Weight: {
                round(observed_data.select('containerWeight').row(index)[0], 3)
            }",
        )
        print(
            f"Active Valves: {
                observed_data.select('activeValveCount').row(index)[0]
            }",
        )
        print(
            f"Initial Pressure: {
                round(observed_data.select('p_initial').row(index)[0], 3)
            }",
        )
        print(
            f"Temperature: {
                round(observed_data.select('T').row(index)[0], 3)
            }",
        )
        print(
            f"Accumulated Pressures: \n{
                observed_data.select('^p_accumulator_.*$').row(index)
            }",
        )


def print_row(observed_data: DataFrame) -> None:
    logger.info("Printing rows interactively:")
    while True:
        index = input("Enter the row index to print or 'x' to quit: ")
        if index == "x":
            break
        print(
            f"Weight: {
                round(
                    observed_data.select('containerWeight').row(int(index))[0],
                    3,
                )
            }",
        )
        print(
            f"Active Valves: {
                observed_data.select('activeValveCount').row(int(index))[0]
            }",
        )
        print(
            f"Initial Pressure: {
                round(observed_data.select('p_initial').row(int(index))[0], 3)
            }",
        )
        print(
            f"Temperature: {
                round(observed_data.select('T').row(int(index))[0], 3)
            }",
        )
        print(
            f"Accumulated Pressures: \n{
                observed_data.select('^p_accumulator_.*$').row(int(index))
            }",
        )


def plot_data(args: argparse.Namespace, observed_data: DataFrame) -> None:
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


def plot_row(observed_data: DataFrame) -> None:
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
    data: DataFrame,
    inputs: list,
    outputs: list,
) -> None:
    logger.info("Training the model:")
    time_start = time.time()

    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
        data,
    )
    if args.use_lightning_learner:
        learner = LightningLearner(
            MultilayerPerceptron(
                learning_rate=args.lightning_learning_rate,
                input_size=len(inputs),
                output_size=len(outputs),
            ),
        )
    else:
        tree_params = {
            "max_depth": args.tree_max_depth,
            "min_samples_split": args.tree_min_samples_split,
            "min_samples_leaf": args.tree_min_samples_leaf,
            "max_leaf_nodes": args.tree_max_leaf_nodes,
            "min_impurity_decrease": args.tree_min_impurity_decrease,
            "ccp_alpha": args.tree_ccp_alpha,
        }
        if args.store_graph:
            Path.mkdir(Path("./graphs"), exist_ok=True)
            tree_path = Path(
                f"./graphs/regression_tree_{time.strftime('%Y%m%d-%H%M%S')}.dot",
            )
            tree_path.open(mode="w").close()
            learner = RegressionTree(
                dot_graph_export_path=str(tree_path),
                **tree_params,
            )
        else:
            learner = RegressionTree(**tree_params)

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


def get_graph(args: argparse.Namespace) -> graphviz.Source:
    graph_path = None

    if args.show_latest_graph:
        list_of_graphs = os.listdir("./graphs")
        if list_of_graphs:
            graph_path = Path("./graphs/" + max(list_of_graphs))
        else:
            logger.warning("No dot-graphs found in './graphs'.")

    elif args.show_graph:
        path = Path("./graphs/" + args.show_graph)
        if path.exists():
            graph_path = path
        else:
            logger.warning(
                "No dot-graph found at './graphs/%s'.", args.show_graph,
            )

    if graph_path:
        with graph_path.open() as file:
            dot_graph = file.read()
        return graphviz.Source(dot_graph)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run the Automatic Lashing Platform example.",
        epilog=(
            """Try this example: uv run run.py --time_window_end 5 """
            """--apply_derivative --filter \'"activeValveCount > 0"\' """
            """--plot_data --plot_distributed --plots 9"""
        ),
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
        "--time_window_start",
        type=float,
        default=0.0,
        metavar="TIME",
        help="Set the start of the time window. Range: [0, 15] (default: 0.0)",
    )
    parameter_group.add_argument(
        "--time_window_end",
        type=float,
        default=15.0,
        metavar="TIME",
        help="Set the end of the time window. Range: [0, 15] (default: 15.0)",
    )
    parameter_group.add_argument(
        "--sample_rate",
        type=float,
        default=0.01,
        metavar="RATE",
        help="Set the sample rate for the data. (default: 0.01 [1500 Values])",
    )
    parameter_group.add_argument(
        "--filter",
        type=str,
        default='"1==1"',
        metavar="CONDITION",
        help=(
            """Filter the data with a condition like """
            """\'And(["activeValveCount > 0", "activeValveCount < 3"])\' """
            """or simple something like \'"activeValveCount > 0"\'."""
        ),
    )
    parameter_group.add_argument(
        "--apply_derivative",
        action="store_true",
        help="Applying the derivative to the data.",
    )
    parameter_group.add_argument(
        "--only_pressure_curve",
        action="store_true",
        help="Use only the pressure curve as input for training.",
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
        "--tree_max_depth",
        type=int,
        default=None,
        metavar="DEPTH",
        help="Set the maximum depth of the regression-tree. (default: None)",
    )
    training_group.add_argument(
        "--tree_min_samples_split",
        type=int,
        default=2,
        metavar="SAMPLES",
        help=(
            "Set the minimum samples to split the regression-tree. "
            "(default: 2)"
        ),
    )
    training_group.add_argument(
        "--tree_min_samples_leaf",
        type=int,
        default=1,
        metavar="SAMPLES",
        help=(
            "Set the minimum samples in a leaf of the regression-tree. "
            "(default: 1)"
        ),
    )
    training_group.add_argument(
        "--tree_max_leaf_nodes",
        type=int,
        default=None,
        metavar="NODES",
        help="Set the maximum leaf nodes of the regression-tree. "
        "(default: None)",
    )
    training_group.add_argument(
        "--tree_min_impurity_decrease",
        type=float,
        default=0.0,
        metavar="DECREASE",
        help=(
            "Set the minimum impurity decrease of the regression-tree. "
            "(default: 0.0)"
        ),
    )
    training_group.add_argument(
        "--tree_ccp_alpha",
        type=float,
        default=0.0,
        metavar="ALPHA",
        help=(
            "Set the complexity parameter of the regression-tree. "
            "(default: 0.0)"
        ),
    )
    training_group.add_argument(
        "--use_lightning_learner",
        action="store_true",
        help=(
            "Use the Lightning Learner with Multilayer-Perceptron "
            "instead of Regression-Tree."
        ),
    )
    training_group.add_argument(
        "--lightning_learning_rate",
        type=float,
        default=0.1,
        metavar="RATE",
        help="Set the learning rate for the lightning-model. (default: 0.1)",
    )

    # training evaluation

    evaluation_group = parser.add_argument_group(
        "Model-Evaluation",
        "Options to evaluate the model.",
    )
    evaluation_group.add_argument(
        "--store_graph",
        action="store_true",
        help="Store the regression-tree as dot-graph at './graphs'.",
    )
    evaluation_group.add_argument(
        "--show_latest_graph",
        action="store_true",
        help="Show the latest stored dot-graph.",
    )
    evaluation_group.add_argument(
        "--show_graph",
        type=str,
        metavar="FILE-NAME",
        help="Show a stored dot-graph.",
    )

    args = parser.parse_args()

    main(args)
