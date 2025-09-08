import logging
import time

import flowcean
import flowcean.cli
from flowcean.core.strategies.offline import evaluate_offline, learn_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.polars.environments.train_test_split import TrainTestSplit
from flowcean.polars.transforms import (
    Filter,
    Flatten,
    Lambda,
    Resample,
    Select,
    TimeWindow,
)
from flowcean.sklearn.metrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
)
from flowcean.xgboost.learner import (
    XGBoostRegressorLearner,
)

logger = logging.getLogger("main")


def main() -> None:
    flowcean.cli.initialize()
    logger.info("Loading data...")

    time_start = time.time()
    data = (
        DataFrame.from_parquet("./data/alp_sim_data_compressed.parquet")
        | Lambda(lambda df: df.limit(50_000))
        | Select(
            [
                "p_accumulator",
                "containerWeight",
                "activeValveCount",
                "T",
            ],
        )
        | Filter("activeValveCount > 0")
        | Resample(0.25)
        | TimeWindow(
            time_start=0,
            time_end=6,
        )
        | Flatten()
    )
    time_end = time.time()
    logger.info("Took %.5f s to load data", time_end - time_start)

    time_start = time.time()
    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=True).split(data)
    time_end = time.time()
    logger.info(
        "Took %.5f s to split data into train and test environments",
        time_end - time_start,
    )

    learner = XGBoostRegressorLearner(
        n_estimators=10,
        max_depth=20,
        objective="reg:squarederror",
    )
    inputs = [
        "^p_accumulator_.*$",
        "activeValveCount",
        "T",
    ]
    outputs = ["containerWeight"]

    logger.info("Starting learning...")
    time_start = time.time()
    model = learn_offline(
        train_env,
        learner,
        inputs,
        outputs,
    )
    time_end = time.time()
    logger.info("Took %.5f s to learn model", time_end - time_start)

    logger.info("Starting evaluation...")
    time_start = time.time()
    report = evaluate_offline(
        model,
        test_env,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    time_end = time.time()
    print(report)
    logger.info("Took %.5f s to evaluate model", time_end - time_start)

    logger.info("Saving model...")
    time_start = time.time()
    model.save("./models/xgboost_regressor_model.fml")
    time_end = time.time()
    logger.info("Took %.5f s to save model", time_end - time_start)


if __name__ == "__main__":
    main()
