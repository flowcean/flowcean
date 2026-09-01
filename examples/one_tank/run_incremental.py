import logging
from datetime import UTC, datetime

import numpy as np
from river import tree
from system import simulate_one_tank

import flowcean.cli
from flowcean.core import evaluate_offline, learn_incremental
from flowcean.polars import (
    DataFrame,
    SlidingWindow,
    StreamingOfflineEnvironment,
    TrainTestSplit,
)
from flowcean.river import RiverLearner
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanSquaredError,
)

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize()

    inputs = ["h_0", "h_1"]
    outputs = ["h_2"]

    data = DataFrame(simulate_one_tank()) | SlidingWindow(window_size=3)

    # Split the data into train and test sets
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    train_incremental = StreamingOfflineEnvironment(train, batch_size=1)

    learner = RiverLearner(
        model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5),
    )

    t_start = datetime.now(tz=UTC)
    model = learn_incremental(
        train_incremental,
        learner,
        inputs,
        outputs,
    )
    delta_t = datetime.now(tz=UTC) - t_start
    print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")

    report = evaluate_offline(
        model,
        test,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    print(report)
    logger.info("Model learning successful.")


if __name__ == "__main__":
    main()
