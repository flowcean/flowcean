import logging

import agenc.cli
from agenc.data.train_test_split import TrainTestSplit
from agenc.data.uri import UriDataLoader
from agenc.learners.regression_tree import RegressionTree
from agenc.metrics.regression import MeanAbsoluteError, MeanSquaredError
from agenc.strategies.offline import evaluate, learn_offline
from agenc.transforms import Select, SlidingWindow, Standardize

logger = logging.getLogger(__name__)


def main() -> None:
    agenc.cli.initialize()

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
