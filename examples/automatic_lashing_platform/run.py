import logging
import time
from pathlib import Path

import polars as pl
from utils.data import split_dataset
from utils.plot import plot_alp_pressures, plot_performances

import flowcean
import flowcean.cli
import flowcean.utils.random
from flowcean.core import evaluate_offline, learn_offline
from flowcean.polars import (
    DataFrame,
)
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    RegressionTree,
)
from flowcean.xgboost import XGBoostRegressorLearner

logger = logging.getLogger(__name__)


def main() -> None:
    _config = flowcean.cli.initialize()
    flowcean.utils.random.initialize_random(42)

    # Keep the dataset split on a fixed seed without advancing learner seeds.
    split_seed = flowcean.utils.random.get_seed()

    logger.info("Loading data...")

    time_start = time.time()
    train_path = Path("./data/alp_sim_data.train.parquet")
    test_path = Path("./data/alp_sim_data.test.parquet")
    if not train_path.exists() or not test_path.exists():
        logger.info("Processed data not found, splitting dataset...")
        split_dataset(seed=split_seed)

    train_env = DataFrame.from_parquet(train_path)
    test_env = DataFrame.from_parquet(test_path)
    time_end = time.time()
    logger.info("Took %.5f s to load data", time_end - time_start)

    feature_combinations = [
        ["^p_accumulator_[0-9]*$"],
        ["^p_accumulator_[0-9]*$", "T", "active_valve_count"],
        ["^p_accumulator_derivative_[0-9]*$"],
        ["p_accumulator_0", "^p_accumulator_derivative_[0-9]*$"],
        ["^p_accumulator_derivative_[0-9]*$", "T", "active_valve_count"],
    ]
    depth = [1, 2, 3, 4, 5, 7, 9, 11, 15, 20, 25, 30, 40]

    for features in feature_combinations:
        results = {
            "depth": [],
            "MAE": [],
            "MSE": [],
            "MAPE": [],
        }
        for d in depth:
            logger.info(
                "Training model with features %s and depth %d",
                features,
                d,
            )
            learner = RegressionTree(
                max_depth=d,
                random_state=flowcean.utils.random.get_seed(),
            )
            model = learn_offline(
                train_env,
                learner,
                features,
                ["container_weight"],
            )
            report = evaluate_offline(
                model,
                test_env,
                features,
                ["container_weight"],
                [
                    MeanAbsoluteError(),
                    MeanSquaredError(),
                    MeanAbsolutePercentageError(),
                ],
            )
            logger.info("Evaluation report:\n%s", report)
            model.save(
                f"./models/regression_tree_features_{'_'.join(features)}_depth_{d}.fml",
            )
            entry = report["DecisionTreeRegressor"].flatten(delimiter="->")

            results["depth"].append(d)
            results["MAE"].append(entry["MeanAbsoluteError->container_weight"])
            results["MSE"].append(entry["MeanSquaredError->container_weight"])
            results["MAPE"].append(
                entry["MeanAbsolutePercentageError->container_weight"],
            )

        pl.DataFrame(results).write_csv(
            f"./results/results_regression_tree_{'_'.join(features)}.csv",
        )

    learner = XGBoostRegressorLearner(
        n_estimators=40,
        max_depth=30,
        learning_rate=0.1,
        random_state=flowcean.utils.random.get_seed(),
    )
    model = learn_offline(
        train_env,
        learner,
        ["^p_accumulator_[0-9]*$"],
        ["container_weight"],
    )
    report = evaluate_offline(
        model,
        test_env,
        ["^p_accumulator_[0-9]*$"],
        ["container_weight"],
        [
            MeanAbsoluteError(),
            MeanSquaredError(),
            MeanAbsolutePercentageError(),
        ],
    )
    logger.info("XGBoost Evaluation report:\n%s", report)
    model.save("./models/xgboost_regressor.fml")

    plot_alp_pressures(weight=45000)
    plot_performances()


if __name__ == "__main__":
    main()
