# system libraries
import argparse
import logging
import math
import time
from os import environ
from pathlib import Path
from sys import base_prefix
from typing import Any

# third-party libraries
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

# flowcean libraries
from transforms.derivative import Derivative

import flowcean.cli
from flowcean.core import SupervisedLearner, evaluate_offline, learn_offline
from flowcean.polars import (
    DataFrame,
    Filter,
    Flatten,
    Resample,
    Select,
    TimeWindow,
    TrainTestSplit,
)
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    RegressionTree,
)
from flowcean.torch import (
    LightningLearner,
    MultilayerPerceptron,
)

# set matplotlib backend Tcl/Tk libraries for the usage in uv
environ["TCL_LIBRARY"] = str(Path(base_prefix) / "tcl" / "tcl8.6")
environ["TK_LIBRARY"] = str(Path(base_prefix) / "tcl" / "tk8.6")

# constants
NATIVE_SAMPLE_RATE = 0.01  # 1500 values per second

# start logger
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    flowcean.cli.initialize(parse_arguments=False)

    data, inputs, outputs = load_and_prepare_data(args)
    inspect_data(args, data)

    if args.train_nodes_vs_error:
        train_nodes_vs_error(args, data, inputs, outputs)
    elif args.train_depth_vs_error:
        train_depth_vs_error(args, data, inputs, outputs)
    elif args.train_time_vs_error:
        train_time_vs_error(args, data, inputs, outputs)
    elif not args.no_training:
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
            plot_row(args, observed_data)


def print_overview(observed_data: DataFrame) -> None:
    logger.info("Data overview:")
    print(observed_data)


def check_redundancy(observed_data: Any) -> None:
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
    observed_data: Any,
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


def print_row(observed_data: Any) -> None:
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


def set_plot_labels() -> None:
    if args.sample_rate == NATIVE_SAMPLE_RATE:
        plt.xlabel("Time [ms]")
    else:
        plt.xlabel("Time [samples]")
    if args.plot_normalized:
        plt.ylabel("Normalized Accumulator Pressure [bar]")
    elif args.apply_derivative:
        plt.ylabel("Derivative of Accumulator Pressure [bar]")
    else:
        plt.ylabel("Accumulator Pressure [bar]")


def set_plot_style(
    args: argparse.Namespace,
    weight: float,
    index: int,
) -> None:
    if args.plot_plain:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
    elif not args.plot_without_notations:
        if args.plot_subplots:
            plt.title(
                f"Weight: {weight}, Index: {index}",
            )
        else:
            set_plot_labels()


def get_pressure_data(
    observed_data: Any,
    index: int,
    *,
    normalized: bool = False,
) -> Any:
    pressure_data = observed_data.select("^p_accumulator_.*$").row(int(index))
    # normalization of the pressure data
    if normalized:
        pressure_data = np.array(pressure_data)
        pressure_data = pressure_data - pressure_data[0]
    return pressure_data


def get_scaled_time(pressure_data: Any, sample_rate: float) -> np.ndarray:
    # Scale x-axis to milliseconds if sample rate is native
    if sample_rate == NATIVE_SAMPLE_RATE:
        return np.arange(len(pressure_data)) * 10  # 10 milliseconds per sample
    return np.arange(len(pressure_data))


def plot_with_legend(
    time: np.ndarray,
    pressure_data: Any,
    weight: float,
    color: str = "",
) -> None:
    if color:
        plt.plot(time, pressure_data, color=color, label=f"{int(weight)} tons")
    else:
        plt.plot(time, pressure_data, label=f"{int(weight)} tons")
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_handles_labels = sorted(
        zip(labels, handles, strict=False),
        key=lambda x: x[0],
    )
    labels, handles = zip(*sorted_handles_labels, strict=False)
    plt.legend(handles, labels)


def plot_data(args: argparse.Namespace, observed_data: Any) -> None:
    logger.info("Plotting %d rows:", args.plots)

    dimension = observed_data.shape[0]
    plots = min(dimension, args.plots)
    plot_rows = math.ceil(math.sqrt(plots))
    plot_cols = math.ceil(plots / plot_rows)

    plt.figure(figsize=(8, 6))

    for i, c in zip(
        range(0, dimension, int(dimension / plots)),
        range(plots),
        strict=False,
    ):
        index = i if args.plot_distributed else c

        pressure_data = get_pressure_data(
            observed_data,
            index,
            normalized=args.plot_normalized,
        )

        weight = round(
            observed_data.select("containerWeight").row(index)[0],
            3,
        )

        time = get_scaled_time(
            pressure_data,
            args.sample_rate,
        )

        # define subplot
        if args.plot_subplots:
            plt.subplot(plot_rows, plot_cols, c + 1)
            plt.subplots_adjust(
                hspace=0.5,
                wspace=0.5,
                left=0.1,
                right=0.95,
                top=0.9,
                bottom=0.1,
            )

        set_plot_style(args, weight, index)

        # plot the data
        epsilon = args.plot_highlight_similar_weight_range / 2
        if args.plot_highlight_similar_weight == 0 or (
            weight < args.plot_highlight_similar_weight + epsilon
            and weight > args.plot_highlight_similar_weight - epsilon
        ):
            if args.plot_legend:
                plot_with_legend(time, pressure_data, weight)
            else:
                plt.plot(time, pressure_data)
        elif args.plot_legend:
            plot_with_legend(time, pressure_data, weight, color="gray")
        else:
            plt.plot(time, pressure_data, color="gray")

    plt.show()


def plot_row(args: argparse.Namespace, observed_data: Any) -> None:
    logger.info("Plotting rows interactively:")

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

    while True:
        index = input("Enter the row index to plot or 'x' to quit: ")
        if index == "x":
            break
        weight = round(
            observed_data.select("containerWeight").row(int(index))[0],
            3,
        )
        active_valves = observed_data.select("activeValveCount").row(
            int(index),
        )[0]
        temperature = round(observed_data.select("T").row(int(index))[0], 3)
        initial_pressure = round(
            observed_data.select("p_initial").row(int(index))[0],
            3,
        )

        pressure_data = get_pressure_data(
            observed_data,
            int(index),
            normalized=args.plot_normalized,
        )

        time = get_scaled_time(
            pressure_data,
            args.sample_rate,
        )

        if args.plot_plain:
            # only plot the plain graph
            plt.plot(pressure_data)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("")
            plt.ylabel("")
        else:
            set_plot_labels()
            plt.plot(time, pressure_data)

            # Table with parameter of the simulation run
            parameter_table = [
                ["Container Weight", f"{weight:.2f} tons"],
                ["Active Valves", f"{active_valves}"],
                ["Temperature", f"{temperature:.2f} °C"],
                ["Initial Pressure", f"{initial_pressure:.2f} bar"],
            ]
            ax = plt.gca()
            if args.apply_derivative:
                table = Table(ax, loc="upper right")
            else:
                table = Table(ax, loc="lower right")
            for i, row in enumerate(parameter_table):
                for j, cell in enumerate(row):
                    table_cell = table.add_cell(
                        i,
                        j,
                        text=cell,
                        loc="left",
                        width=0.3,
                        height=0.075,
                    )
                    table_cell.get_text().set_fontfamily("serif")
                    table_cell.get_text().set_fontsize(10)
                    table_cell.set_facecolor("white")
            ax.add_table(table)

            plt.grid(visible=True, linestyle="--", alpha=0.5)
        plt.show()


def build_tree_learner(
    args: argparse.Namespace,
    **tree_params: Any,
) -> SupervisedLearner:
    if args.store_graph:
        Path.mkdir(Path("./graphs"), exist_ok=True)
        tree_path = Path(
            f"./graphs/regression_tree_{time.strftime('%Y%m%d-%H%M%S')}.dot",
        )
        tree_path.open(mode="w").close()
        return RegressionTree(
            dot_graph_export_path=str(tree_path),
            **tree_params,
        )
    return RegressionTree(**tree_params)


def train_nodes_vs_error(
    args: argparse.Namespace,
    data: DataFrame,
    inputs: list,
    outputs: list,
) -> None:
    logger.info(
        "Training the model with %d as maximum number of nodes:",
        args.train_nodes_vs_error_max_nodes,
    )

    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
        data,
    )

    errors = []
    node_numbers = []
    for nodes in range(
        2,
        args.train_nodes_vs_error_max_nodes,
        args.train_nodes_vs_error_steps,
    ):
        logger.info("Training with %d nodes:", nodes)
        time_start = time.time()

        tree_params = {
            "max_leaf_nodes": nodes,
        }
        learner = build_tree_learner(args, **tree_params)
        model = learn_offline(train_env, learner, inputs, outputs)
        report = evaluate_offline(
            model,
            test_env,
            inputs,
            outputs,
            [MeanSquaredError()],
        )

        time_after_learning = time.time()
        logger.info(
            "Took %.5f s to learn model",
            time_after_learning - time_start,
        )
        errors.append(report["MeanSquaredError"])
        node_numbers.append(nodes)

    # calculate optimal number of nodes
    distances = np.sqrt(np.array(node_numbers) ** 2 + np.array(errors) ** 2)
    min_index = np.argmin(distances)
    logger.info(
        "Optimal number of nodes: %d with error: %.2f",
        node_numbers[min_index],
        errors[min_index],
    )

    # plot errors
    plt.figure()
    plt.plot(node_numbers, errors)
    plt.plot(node_numbers[min_index], errors[min_index], "ro", markersize=6)
    plt.annotate(
        f"({node_numbers[min_index]}, {errors[min_index]:.2f})",
        xy=(node_numbers[min_index], errors[min_index]),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.3",
            "edgecolor": "gray",
            "facecolor": "white",
        },
    )
    plt.xlabel("Number of Leaf Nodes")
    plt.ylabel("Mean Squared Error")
    plt.title("Leaf Nodes vs. Error")
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.show()


def train_depth_vs_error(
    args: argparse.Namespace,
    data: DataFrame,
    inputs: list,
    outputs: list,
) -> None:
    logger.info(
        "Training the model with %d as maximum depth of the tree:",
        args.train_depth_vs_error_max_depth,
    )

    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
        data,
    )

    errors = []
    depth_numbers = []
    for depth in range(
        2,
        args.train_depth_vs_error_max_depth,
        args.train_depth_vs_error_steps,
    ):
        logger.info("Training with a depth of %d:", depth)
        time_start = time.time()

        tree_params = {
            "max_depth": depth,
        }
        learner: Any = build_tree_learner(args, **tree_params)
        model = learn_offline(train_env, learner, inputs, outputs)
        report = evaluate_offline(
            model,
            test_env,
            inputs,
            outputs,
            [MeanSquaredError()],
        )

        time_after_learning = time.time()
        logger.info(
            "Took %.5f s to learn model",
            time_after_learning - time_start,
        )
        errors.append(report["MeanSquaredError"])
        depth_numbers.append(depth)

        if learner.regressor.get_depth() < depth:
            logger.info(
                "Max depth of the model is less than %d.",
                depth,
            )
            break

    # calculate optimal number of nodes
    distances = np.sqrt(np.array(depth_numbers) ** 2 + np.array(errors) ** 2)
    min_index = np.argmin(distances)
    logger.info(
        "Optimal number of depth: %d with error: %.2f",
        depth_numbers[min_index],
        errors[min_index],
    )

    # plot errors
    plt.plot(depth_numbers, errors)
    plt.plot(depth_numbers[min_index], errors[min_index], "ro", markersize=6)
    plt.annotate(
        f"({depth_numbers[min_index]}, {errors[min_index]:.2f})",
        xy=(depth_numbers[min_index], errors[min_index]),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.3",
            "edgecolor": "gray",
            "facecolor": "white",
        },
    )
    plt.xlabel("Depth of the Tree")
    plt.ylabel("Mean Squared Error")
    plt.title("Tree Depth vs. Error")
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.show()


def train_time_vs_error(
    args: argparse.Namespace,
    data: DataFrame,
    inputs: list,
    outputs: list,
) -> None:
    logger.info(
        "Training the model with %d as maximum seconds to train the model",
        args.train_time_vs_error_max_time,
    )

    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
        data,
    )

    errors = []
    times = []
    depths = []
    depth_count = 1
    while True:
        logger.info("Training with %d as max-depth:", depth_count)
        time_start = time.time()

        tree_params = {
            "max_depth": depth_count,
        }
        learner: Any = build_tree_learner(args, **tree_params)
        model = learn_offline(train_env, learner, inputs, outputs)
        report = evaluate_offline(
            model,
            test_env,
            inputs,
            outputs,
            [MeanSquaredError()],
        )

        time_to_learn = time.time() - time_start
        logger.info(
            "Took %.5f s to learn model",
            time_to_learn,
        )
        errors.append(report["MeanSquaredError"])
        times.append(time_to_learn)
        depths.append(depth_count)

        if time_to_learn > args.train_time_vs_error_max_time:
            logger.info(
                "Time to train the model exceeded %d seconds.",
                args.train_time_vs_error_max_time,
            )
            break
        if learner.regressor.get_depth() < depth_count:
            logger.info(
                "Max depth of the model is less than %d.",
                depth_count,
            )
            break
        depth_count += 1

    # calculate optimal number of nodes
    distances = np.sqrt(np.array(times) ** 2 + np.array(errors) ** 2)
    min_index = np.argmin(distances)
    logger.info(
        "Optimal time to train: %.2f with error: %.2f and depth: %d",
        times[min_index],
        errors[min_index],
        depths[min_index],
    )

    # plot errors
    plt.figure()
    plt.plot(times, errors)
    plt.plot(times[min_index], errors[min_index], "ro", markersize=6)
    plt.annotate(
        f"({times[min_index]:.2f}, {errors[min_index]:.2f})\n"
        f"Depth: {depths[min_index]}",
        xy=(times[min_index], errors[min_index]),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.3",
            "edgecolor": "gray",
            "facecolor": "white",
        },
    )
    plt.xlabel("Time to Train (s)")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Duration vs. Error")
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.show()


def train_and_evaluate_model(
    args: argparse.Namespace,
    data: DataFrame,
    inputs: list,
    outputs: list,
) -> None:
    logger.info("Training the model:")

    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
        data,
    )

    time_start = time.time()

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
        learner = build_tree_learner(args, **tree_params)

    model = learn_offline(
        train_env,
        learner,
        inputs,
        outputs,
    )

    if args.store_model:
        Path.mkdir(Path("./models"), exist_ok=True)
        model_path = Path(
            f"./models/model_{time.strftime('%Y%m%d-%H%M%S')}.fml",
        )
        with model_path.open("wb") as f:
            model.save(f)

    report = evaluate_offline(
        model,
        test_env,
        inputs,
        outputs,
        [
            MeanAbsoluteError(),
            MeanSquaredError(),
            MeanAbsolutePercentageError(),
        ],
    )
    time_after_learning = time.time()
    logger.info(
        "Took %.5f s to learn model",
        time_after_learning - time_start,
    )

    print(report)


def get_graph(args: argparse.Namespace) -> graphviz.Source | None:
    graph_path = None

    if args.show_latest_graph:
        graph_dir = Path("./graphs")
        graph_files = [f for f in graph_dir.iterdir() if f.suffix == ".dot"]
        if graph_files:
            latest_graph = max(graph_files, key=lambda p: p.stat().st_mtime)
            graph_path = latest_graph
        else:
            logger.warning("No dot-graphs found in './graphs'.")

    elif args.show_graph:
        path = Path("./graphs/" + args.show_graph)
        if path.exists():
            graph_path = path
        else:
            logger.warning(
                "No dot-graph found at './graphs/%s'.",
                args.show_graph,
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
            """Try this example for learning with best-known parameters: """
            """uv run run.py --time_window_end 4 """
            """--apply_derivative --filter \'"activeValveCount > 0"\' """
            """--tree_min_impurity_decrease 0.1 --tree_max_depth 10 """
            """--plot_data --plot_distributed --plots 9 """
            """--store_graph --show_latest_graph"""
        ),
    )

    # parameter

    parameter_group = parser.add_argument_group(
        "Parameters",
        "Parameter options for the training data.",
    )
    parameter_group.add_argument(
        "--training_data",
        type=str,
        default="alp_sim_data.parquet",
        metavar="FILE",
        help="Set the training data file. (default: alp_sim_data.parquet)",
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
            """or simply something like \'"activeValveCount > 0"\'."""
        ),
    )
    parameter_group.add_argument(
        "--apply_derivative",
        action="store_true",
        help="Apply the derivative to the data.",
    )
    parameter_group.add_argument(
        "--only_pressure_curve",
        action="store_true",
        help="Use only the pressure curve as input for training.",
    )

    # training data inspection

    data_inspection_group = parser.add_argument_group(
        "Training Data",
        "Tools to inspect the training data.",
    )
    data_inspection_group.add_argument(
        "--print_overview",
        action="store_true",
        help="Print a short overview of the training data.",
    )
    data_inspection_group.add_argument(
        "--check_redundancy",
        action="store_true",
        help=(
            "Check for duplicated and unique output values in the "
            "training data. (only for containerWeight)"
        ),
    )
    data_inspection_group.add_argument(
        "--print_data",
        action="store_true",
        help="Print a number of rows of the training data.",
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
        help="Print from distributed training data.",
    )
    data_inspection_group.add_argument(
        "--print_row",
        action="store_true",
        help="Print a row of the training data interactively.",
    )
    data_inspection_group.add_argument(
        "--plot_data",
        action="store_true",
        help="Plot the training data.",
    )
    data_inspection_group.add_argument(
        "--plots",
        type=int,
        default=20,
        metavar="NUMBER",
        help="Number of plots to show. (default: 20)",
    )
    data_inspection_group.add_argument(
        "--plot_plain",
        action="store_true",
        help="Plot only the graph without notations and axis labels.",
    )
    data_inspection_group.add_argument(
        "--plot_legend",
        action="store_true",
        help=(
            "Plot the legend with the weight of the container in the graph."
        ),
    )
    data_inspection_group.add_argument(
        "--plot_without_notations",
        action="store_true",
        help="Plot the graph without notations. (for --plot_data only)",
    )
    data_inspection_group.add_argument(
        "--plot_subplots",
        action="store_true",
        help="Plot the graphs as subplots. (for --plot_data only)",
    )
    data_inspection_group.add_argument(
        "--plot_normalized",
        action="store_true",
        help="Plot the pressure data normalized to the first value.",
    )
    data_inspection_group.add_argument(
        "--plot_highlight_similar_weight",
        type=int,
        default=0,
        metavar="WEIGHT [tons]",
        help=(
            "Plot only those curves colorfully which correspond "
            "to the given weight in range. "
            "(for --plot_data only)"
        ),
    )
    data_inspection_group.add_argument(
        "--plot_highlight_similar_weight_range",
        type=int,
        default=10,
        metavar="RANGE [tons]",
        help=(
            "Range around the weight to highlight in the plot. "
            "(default: 10, meaning the weight is in the range of "
            "marked_weight ± range/2)"
        ),
    )
    data_inspection_group.add_argument(
        "--plot_distributed",
        action="store_true",
        help="Plot from distributed training data.",
    )
    data_inspection_group.add_argument(
        "--plot_row",
        action="store_true",
        help="Plot a row of the training data interactively.",
    )

    # training

    training_group = parser.add_argument_group(
        "Model Training",
        "Options to train the model.",
    )
    training_group.add_argument(
        "--no_training",
        action="store_true",
        help="Do not train.",
    )
    training_group.add_argument(
        "--train_nodes_vs_error",
        action="store_true",
        help=(
            "Train the model with different numbers of nodes, plot the "
            "error and give the optimal number of nodes."
        ),
    )
    training_group.add_argument(
        "--train_nodes_vs_error_max_nodes",
        type=int,
        default=16,
        metavar="NODES",
        help=(
            "Set the maximum number of leaf nodes for the training. "
            "(default: 16)"
        ),
    )
    training_group.add_argument(
        "--train_nodes_vs_error_steps",
        type=int,
        default=2,
        metavar="STEPS",
        help=(
            "Set the number of nodes to increment after each iteration. "
            "(default: 2)"
        ),
    )
    training_group.add_argument(
        "--train_depth_vs_error",
        action="store_true",
        help=(
            "Train the model with different tree depths, plot the "
            "error and give the optimal depth."
        ),
    )
    training_group.add_argument(
        "--train_depth_vs_error_max_depth",
        type=int,
        default=20,
        metavar="DEPTH",
        help="Set the maximum depth of the regression tree. (default: 20)",
    )
    training_group.add_argument(
        "--train_depth_vs_error_steps",
        type=int,
        default=1,
        metavar="STEPS",
        help=(
            "Set the number of depth increments after each iteration. "
            "(default: 1)"
        ),
    )
    training_group.add_argument(
        "--train_time_vs_error",
        action="store_true",
        help=(
            "Train the model with different training times, plot the error "
            "and give the optimal time and nodes to train."
        ),
    )
    training_group.add_argument(
        "--train_time_vs_error_max_time",
        type=int,
        default=8,
        metavar="TIME",
        help=(
            "Set the maximum time (seconds) to train the model. (default: 8)"
        ),
    )
    training_group.add_argument(
        "--tree_max_depth",
        type=int,
        default=None,
        metavar="DEPTH",
        help="Set the maximum depth of the regression tree. (default: None)",
    )
    training_group.add_argument(
        "--tree_min_samples_split",
        type=int,
        default=2,
        metavar="SAMPLES",
        help=(
            "Set the minimum samples to split the regression tree. "
            "(default: 2)"
        ),
    )
    training_group.add_argument(
        "--tree_min_samples_leaf",
        type=int,
        default=1,
        metavar="SAMPLES",
        help=(
            "Set the minimum samples in a leaf of the regression tree. "
            "(default: 1)"
        ),
    )
    training_group.add_argument(
        "--tree_max_leaf_nodes",
        type=int,
        default=None,
        metavar="NODES",
        help="Set the maximum leaf nodes of the regression tree. "
        "(default: None)",
    )
    training_group.add_argument(
        "--tree_min_impurity_decrease",
        type=float,
        default=0.0,
        metavar="DECREASE",
        help=(
            "Set the minimum impurity decrease of the regression tree. "
            "(default: 0.0)"
        ),
    )
    training_group.add_argument(
        "--tree_ccp_alpha",
        type=float,
        default=0.0,
        metavar="ALPHA",
        help=(
            "Set the complexity parameter of the regression tree. "
            "(default: 0.0)"
        ),
    )
    training_group.add_argument(
        "--use_lightning_learner",
        action="store_true",
        help=(
            "Use the Lightning Learner with Multilayer Perceptron "
            "instead of Regression Tree."
        ),
    )
    training_group.add_argument(
        "--lightning_learning_rate",
        type=float,
        default=0.1,
        metavar="RATE",
        help="Set the learning rate for the lightning model. (default: 0.1)",
    )

    # training evaluation

    evaluation_group = parser.add_argument_group(
        "Model Evaluation",
        "Options to evaluate the model.",
    )
    evaluation_group.add_argument(
        "--store_graph",
        action="store_true",
        help="Store the regression tree as a dot-graph at './graphs'.",
    )
    evaluation_group.add_argument(
        "--store_model",
        action="store_true",
        help="Store the trained model as a fml file at './models'.",
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
