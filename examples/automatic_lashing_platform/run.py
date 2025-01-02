#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
#     "matplotlib",
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
from flowcean.transforms import Flatten, Resample, Select

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
        | Resample(args.sample_rate)
        | Flatten()
    )
    time_end = time.time()
    logger.info("Took %.5f s to load data", time_end - time_start)

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


    # print overview of the data

    if args.print_overview:
        logger.info("Data overview:")

        print(observed_data)
        
    
    # print data

    if args.print_data:
        logger.info(f"Printing {args.prints} rows:")

        dimension = observed_data.shape[0]
        prints = args.prints if args.prints < dimension else dimension

        for i, c in zip(range(0, dimension, int(dimension/prints)), range(prints)):
            index = i if args.print_distributed else c
            print(f"Weight: {round(observed_data.select('containerWeight').row(index)[0], 3)}, Index: {index}")
            print(observed_data.select('^p_accumulator_.*$').row(index))


    # print a row of the data interactively

    if args.print_row:
        logger.info(f"Printing rows interactively:")
        while True:
            row_index = input("Enter the row index to print or 'x' to quit: ")
            if row_index == 'x':
                break
            print(f"Weight: {round(observed_data.select('containerWeight').row(int(row_index))[0], 3)}")
            print(observed_data.select('^p_accumulator_.*$').row(int(row_index)))


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
            plt.title(f"Weight: {round(observed_data.select('containerWeight').row(index)[0], 3)}, Index: {index}")
            plt.plot(observed_data.select('^p_accumulator_.*$').row(index))

        plt.subplots_adjust(hspace=0.5, wspace=0.5, left= 0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()


    # plot a row of the data
    if args.plot_row:
        logger.info(f"Plotting rows interactively:")

        plt.figure()
        plt.subplots_adjust(hspace=0.5, wspace=0.5, left= 0.05, right=0.95, top=0.95, bottom=0.05)

        while True:
            row_index = input("Enter the row index to plot or 'x' to quit: ")
            if row_index == 'x':
                break
            plt.title(f"Weight: {round(observed_data.select('containerWeight').row(int(row_index))[0], 3)}, Index: {row_index}")
            plt.plot(observed_data.select('^p_accumulator_.*$').row(int(row_index)))
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run the Automatic Lashing Platform example.",
    )


    # parameter

    parser.add_argument('--sample_rate', type=float, default=1.0, help='Set the sample rate for the data. (default: 1.0) -> 15 Values')


    # data inspection

    parser.add_argument('--print_overview', action='store_true', help='Print a short overview ot the training-data.')
    parser.add_argument('--print_data', action='store_true', help='Print a number of rows of the training-data.')
    parser.add_argument('--prints', type=int, default=10, metavar='NUMBER', help='Number of rows to print. (default: 10)')
    parser.add_argument('--print_distributed', action='store_true', help='Print from distributed training-data.')
    parser.add_argument('--print_row', action='store_true', help='Print a row of the training-data interactively.')
    parser.add_argument('--plot_data', action='store_true', help='Plot the training-data .')
    parser.add_argument('--plots', type=int, default=10, metavar='NUMBER', help='Number of plots to show. (default: 10)')
    parser.add_argument('--plot_distributed', action='store_true', help='Plot from distributed training-data.')
    parser.add_argument('--plot_row', action='store_true', help='Plot a row of the training-data interactively.')


    # training

    parser.add_argument('--no_training', action='store_true', help='Apply no training.')
    parser.add_argument('--lightning_learner', action='store_true', help='Use the Lightning Learner with Multilayer-Perceptron instead of Regression-Tree.')
    parser.add_argument('--learning_rate', type=float, default=0.1, metavar='RATE', help='Set the learning rate for the lightning-model. (default: 0.1)')
    parser.add_argument('--plot_error', action='store_true', help='Plot the error of the trained model.')
    

    args = parser.parse_args()

    main(args)