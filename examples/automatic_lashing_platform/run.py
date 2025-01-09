#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
# ///


# system libraries
import logging
import time
import argparse
import math

# flowcean libraries
import flowcean.cli
from flowcean.environments.parquet import ParquetDataLoader
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.regression_tree import RegressionTree
from flowcean.learners.lightning import LightningLearner, MultilayerPerceptron
from flowcean.metrics.regression import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms import Flatten, Resample, Select #, Filter

# third-party libraries
import matplotlib.pyplot as plt

# start logger
logger = logging.getLogger(__name__)



# method to tain the model for the automatic lashing platform
def main(args) -> None:
    flowcean.cli.initialize_logging(parse_arguments=False)


    # load and prepare training data

    logger.info("Loading data...")
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
        # | Filter(lambda df: df["activeValveCount"] > 0)
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


    # observe data

    logger.info("Observing data...")
    observed_data = data.observe()
    time_after_observe = time.time()
    logger.info("Took %.5f s to observe data", time_after_observe - time_start)


    # print overview of the data

    if args.print_overview:
        logger.info("Data overview:")

        print(observed_data)

    
    # look for redundant output-values in the data

    if args.check_redundancy:
        logger.info("Checking for duplicated and unique output-values:")
        rows = {}
        duplicates = {}
        uniques = {}
        for row, i in zip(observed_data.select("containerWeight").iter_rows(), range(observed_data.select("containerWeight").shape[0])):
            if row[0] in rows:
                if row[0] in duplicates:
                    duplicates[row[0]].append(i)
                else:
                    duplicates[row[0]] = [rows[row[0]], i]
                if row[0] in uniques: uniques.pop(row[0])
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
        
    
    # print data

    if args.print_data:
        logger.info(f"Printing {args.prints} rows:")

        dimension = observed_data.shape[0]
        prints = args.prints if args.prints < dimension else dimension

        for i, c in zip(range(0, dimension, int(dimension/prints)), range(prints)):
            index = i if args.print_distributed else c
            print(f"Weight: {round(observed_data.select('containerWeight').row(index)[0], 3)}, Index: {index}")
            print(f"Active Valves: {observed_data.select('activeValveCount').row(index)[0]}")
            print(f"Initial Pressure: {round(observed_data.select('p_initial').row(index)[0], 3)}")
            print(f"Temperature: {round(observed_data.select('T').row(index)[0], 3)}")
            print(observed_data.select('^p_accumulator_.*$').row(index))


    # print a row of the data interactively

    if args.print_row:
        logger.info(f"Printing rows interactively:")
        while True:
            index = input("Enter the row index to print or 'x' to quit: ")
            if index == 'x':
                break
            print(f"Weight: {round(observed_data.select('containerWeight').row(int(index))[0], 3)}")
            print(f"Active Valves: {observed_data.select('activeValveCount').row(int(index))[0]}")
            print(f"Initial Pressure: {round(observed_data.select('p_initial').row(int(index))[0], 3)}")
            print(f"Temperature: {round(observed_data.select('T').row(int(index))[0], 3)}")
            print(observed_data.select('^p_accumulator_.*$').row(int(index)))


    # plot data

    if args.plot_data:
        logger.info(f"Plotting {args.plots} rows:")

        dimension = observed_data.shape[0]
        plots = args.plots if args.plots < dimension else dimension
        plot_rows = math.ceil(math.sqrt(plots))
        plot_cols = math.ceil(plots / plot_rows)

        plt.figure()
        for i, c in zip(range(0, dimension, int(dimension/plots)), range(plots)):
            plt.subplot(plot_rows, plot_cols, c+1)
            index = i if args.plot_distributed else c
            plt.title(f"Weight: {round(observed_data.select('containerWeight').row(index)[0], 3)}, Active Valves: {observed_data.select('activeValveCount').row(index)[0]}, Index: {index}")
            plt.plot(observed_data.select('^p_accumulator_.*$').row(index))

        plt.subplots_adjust(hspace=0.5, wspace=0.5, left= 0.1, right=0.95, top=0.9, bottom=0.1)
        plt.show()


    # plot a row of the data
    if args.plot_row:
        logger.info(f"Plotting rows interactively:")

        plt.figure()
        plt.subplots_adjust(left= 0.1, right=0.95, top=0.9, bottom=0.1)

        while True:
            index = input("Enter the row index to plot or 'x' to quit: ")
            if index == 'x':
                break
            plt.title(f"Weight: {round(observed_data.select('containerWeight').row(int(index))[0], 3)}, Index: {index}")
            plt.plot(observed_data.select('^p_accumulator_.*$').row(int(index)))
            plt.show()


    # train the model

    if not args.no_training:
        logger.info("Training the model:")

        train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(data)
        time_end = time.time()

        if args.lightning_learner:
            # TODO: test the lightning learner and inspect learning_rate
            learner = LightningLearner(
                MultilayerPerceptron(
                    learning_rate=args.learning_rate, 
                    input_size=len(inputs), 
                    output_size=len(outputs)
                    )
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

        print(report)


        # plot the error of the trained model

        if args.plot_error:
            # TODO
            pass


# parse arguments and run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run the Automatic Lashing Platform example.",
    )


    # parameter

    parameter_group = parser.add_argument_group('Parameter', 'Parameter options for the training-data.')
    parameter_group.add_argument('--sample_rate', type=float, default=1.0, metavar='RATE', help='Set the sample rate for the data. (default: 1.0) -> 15 Values')


    # training-data inspection

    data_inspection_group = parser.add_argument_group('Training-Data', 'Tools to inspect the training-data.')
    data_inspection_group.add_argument('--print_overview', action='store_true', help='Print a short overview ot the training-data.')
    data_inspection_group.add_argument('--check_redundancy', action='store_true', help='Checking for duplicated and unique output-values in the training-data. (only for containerWeight)')
    data_inspection_group.add_argument('--print_data', action='store_true', help='Print a number of rows of the training-data.')
    data_inspection_group.add_argument('--prints', type=int, default=10, metavar='NUMBER', help='Number of rows to print. (default: 10)')
    data_inspection_group.add_argument('--print_distributed', action='store_true', help='Print from distributed training-data.')
    data_inspection_group.add_argument('--print_row', action='store_true', help='Print a row of the training-data interactively.')
    data_inspection_group.add_argument('--plot_data', action='store_true', help='Plot the training-data.')
    data_inspection_group.add_argument('--plots', type=int, default=20, metavar='NUMBER', help='Number of plots to show. (default: 20)')
    data_inspection_group.add_argument('--plot_distributed', action='store_true', help='Plot from distributed training-data.')
    data_inspection_group.add_argument('--plot_row', action='store_true', help='Plot a row of the training-data interactively.')


    # training

    training_group = parser.add_argument_group('Model-Training', 'Options to train the model.')
    training_group.add_argument('--no_training', action='store_true', help='Apply no training.')
    training_group.add_argument('--lightning_learner', action='store_true', help='Use the Lightning Learner with Multilayer-Perceptron instead of Regression-Tree.')
    training_group.add_argument('--learning_rate', type=float, default=0.1, metavar='RATE', help='Set the learning rate for the lightning-model. (default: 0.1)')
    training_group.add_argument('--plot_error', action='store_true', help='Plot the error of the trained model.')
    

    args = parser.parse_args()

    main(args)