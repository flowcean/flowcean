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
import code

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
    # load and prepare training data

    flowcean.cli.initialize_logging(parse_arguments=False)
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

    inputs = [
        "^p_accumulator_.*$",
        "activeValveCount",
        "p_initial",
        "T",
    ]
    outputs = ["containerWeight"]


    # show the training-data

    if args.print_data:
        print(data.observe())

    if args.plot_data:
        observed_data = data.observe()
        # TODO: plot the data
        # code.interact(local=locals())
        # for input in inputs:
        #     plt.plot(observed_data[input], label=input)
        # plt.show()


    # train the model

    if not args.no_training:
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

        if args.plot_error:
            pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run the Automatic Lashing Platform example.",
    )
    parser.add_argument('--print_data', action='store_true', help='Print the data from the parquet file.')
    parser.add_argument('--plot_data', action='store_true', help='Plot the data from the parquet file.')
    parser.add_argument('--no_training', action='store_true', help='Apply no training.')
    parser.add_argument('--lightning_learner', action='store_true', help='Use the Lightning Learner with Multilayer-Perceptron instead of Regression-Tree.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Set the learning rate for the lightning-model. (default: 0.1)')
    parser.add_argument('--plot_error', action='store_true', help='Plot the error of the trained model.')
    
    args = parser.parse_args()

    main(args)