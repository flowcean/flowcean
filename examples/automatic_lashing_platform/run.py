import logging
import time

import polars as pl

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

    logger.info("Loading data...")

    time_start = time.time()
    train_env = DataFrame.from_parquet("./data/alp_sim_data.train.parquet")
    test_env = DataFrame.from_parquet("./data/alp_sim_data.test.parquet")
    time_end = time.time()
    logger.info("Took %.5f s to load data", time_end - time_start)

    # feature_combinations = [
    #     ["^p_accumulator_[0-9]*$"],
    #     ["^p_accumulator_[0-9]*$", "T", "active_valve_count"],
    #     ["^p_accumulator_derivative_[0-9]*$"],
    #     ["p_accumulator_0", "^p_accumulator_derivative_[0-9]*$"],
    #     ["^p_accumulator_derivative_[0-9]*$", "T", "active_valve_count"],
    # ]
    # depth = [1, 2, 3, 4, 5, 7, 9, 11, 15, 20, 25, 30, 40]
    #
    # for features in feature_combinations:
    #     results = {
    #         # "features": [],
    #         "depth": [],
    #         "MAE": [],
    #         "MSE": [],
    #         "MAPE": [],
    #     }
    #     for d in depth:
    #         logger.info(
    #             "Training model with features %s and depth %d",
    #             features,
    #             d,
    #         )
    #         learner = RegressionTree(
    #             max_depth=d,
    #         )
    #         model = learn_offline(
    #             train_env,
    #             learner,
    #             features,
    #             ["container_weight"],
    #         )
    #         report = evaluate_offline(
    #             model,
    #             test_env,
    #             features,
    #             ["container_weight"],
    #             [
    #                 MeanAbsoluteError(),
    #                 MeanSquaredError(),
    #                 MeanAbsolutePercentageError(),
    #             ],
    #         )
    #         logger.info("Evaluation report:\n%s", report)
    #         model.save(
    #             f"./models/regression_tree_features_{'_'.join(features)}_depth_{d}.fml",
    #         )
    #         # results["features"].append(", ".join(features))
    #         results["depth"].append(d)
    #         results["MAE"].append(
    #             report["DecisionTreeRegressor"]["MeanAbsoluteError"][
    #                 "container_weight"
    #             ],
    #         )
    #         results["MSE"].append(
    #             report["DecisionTreeRegressor"]["MeanSquaredError"][
    #                 "container_weight"
    #             ],
    #         )
    #         results["MAPE"].append(
    #             report["DecisionTreeRegressor"]["MeanAbsolutePercentageError"][
    #                 "container_weight"
    #             ],
    #         )
    #
    #     pl.DataFrame(results).write_csv(
    #         f"results_regression_tree_{'_'.join(features)}.csv",
    #     )

    learner = XGBoostRegressorLearner(
        n_estimators=40,
        max_depth=30,
        learning_rate=0.1,
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


if __name__ == "__main__":
    main()
