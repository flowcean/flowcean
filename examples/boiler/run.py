import logging

import flowcean.cli
from flowcean.data.train_test_split import TrainTestSplit
from flowcean.data.uri import UriDataLoader
from flowcean.learners.regression_tree import RegressionTree
from flowcean.metrics import MeanAbsoluteError, MeanSquaredError, evaluate
from flowcean.strategies.offline import learn_offline
from flowcean.transforms import Select, SlidingWindow, Standardize

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()

    data = UriDataLoader(
        uri="file:./data/trace_287401a5.csv",
    ).with_transform(
        Select(features=["reference", "temperature"])
        | SlidingWindow(window_size=3),
    )
    data.load()
    data.get_data()
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    input_transform = Standardize()
    learner = RegressionTree()
    inputs = [
        "reference_0",
        "temperature_0",
        "reference_1",
        "temperature_1",
        "reference_2",
    ]
    outputs = ["temperature_2"]
    model = learn_offline(
        train,
        learner,
        inputs,
        outputs,
        input_transform=input_transform,
    )

    logger.info("Evaluating model")
    report = evaluate(
        model,
        test,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    print(report)


if __name__ == "__main__":
    main()
