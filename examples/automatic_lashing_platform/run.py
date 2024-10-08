import logging
import time
from pathlib import Path

import flowcean.cli
from flowcean.environments.parquet import ParquetDataLoader
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.regression_tree import RegressionTree
from flowcean.metrics import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms import Flatten, Resample, Select
from flowcean.transforms.rechunk import Rechunk

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()
    time_start = time.time()
    data = ParquetDataLoader(Path("./alp_sim_data.parquet")).with_transform(
        Select(
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
        | Rechunk()
    )
    data.load()
    time_end = time.time()
    logger.info("took %.5f s to load data", time_end - time_start)

    time_start = time.time()
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)
    time_end = time.time()
    logger.info("took %.5f s to process data", time_end - time_start)

    learner = RegressionTree()
    inputs = [
        "^p_accumulator_.*$",
        "activeValveCount",
        "p_initial",
        "T",
    ]
    outputs = ["containerWeight"]

    model = learn_offline(
        train,
        learner,
        inputs,
        outputs,
    )

    report = evaluate_offline(
        model,
        test,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    print(report)


if __name__ == "__main__":
    main()
