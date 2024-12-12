#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
#     "matplotlib",
# ]
# ///


# system libraries
import sys
import logging
import time

# flowcean libraries
import flowcean.cli
from flowcean.environments.parquet import ParquetDataLoader
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.regression_tree import RegressionTree
from flowcean.metrics.regression import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms import Flatten, Resample, Select

# third-party libraries
import matplotlib.pyplot as plt
import polars

# start logger
logger = logging.getLogger(__name__)


# method to tain the model for the automatic lashing platform
def main(flags) -> None:
    flowcean.cli.initialize_logging()
    time_start = time.time()

    data = (
        ParquetDataLoader("./alp_sim_data.parquet")
        | Select(
            [
                "p_accumulator",
                "containerWeight",
                "p_initial",
                "activeValveCount",
                "T",
            ]
        )
        | Resample(1.0)
        | Flatten()
    )
    time_end = time.time()
    logger.info("took %.5f s to load data", time_end - time_start)

    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(data)
    time_end = time.time()

    if flags["--print_data"]:
        print(data)

    if flags["--plot_data"]:
        pass

    if not flags["--no_training"]:
        learner = RegressionTree()
        inputs = [
            "^p_accumulator_.*$",
            "activeValveCount",
            "p_initial",
            "T",
        ]
        outputs = ["containerWeight"]

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
        print(report)


if __name__ == "__main__":
    flags = {
        "--print_data": False,
        "--plot_data": False,
        "--no_training": False,
        "--plot_error": False
    }

    if len(sys.argv) > 1:
        for arg in sys.argv:
            for flag in flags:
                print(arg, flag)
                if arg == flag:
                    print("setting flag")
                    flags[flag] = True
            if arg == "--help" or arg == "-h":
                print("usage: run.py [options]")
                print("options:")
                print("\t -h, --help\t\t Show this help message and exit.")
                print("\t --print_data\t Print the data from the parquet file.")
                print("\t --plot_data\t Plot the data from the parquet file.")
                print("\t --no_training\t Apply no training.")
                print("\t --plot_error\t Plot the error of the trained model.")
                sys.exit(0)
    
    print(flags)

    main(flags)