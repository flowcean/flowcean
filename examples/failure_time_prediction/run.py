import logging

import agenc.cli
from agenc.data.train_test_split import TrainTestSplit
from agenc.data.uri import UriDataLoader
from agenc.learners.lightning import LightningLearner, MultilayerPerceptron
from agenc.metrics.regression import MeanAbsoluteError, MeanSquaredError
from agenc.strategies.offline import evaluate, learn_offline
from agenc.transforms import Select, Standardize

logger = logging.getLogger(__name__)


def main() -> None:
    agenc.cli.initialize()

    data = UriDataLoader(uri="file:./data/processed_data.csv").with_transform(
        Select(
            [
                "y-Amplitude",
                "z-Amplitude",
                "Growth-rate",
                "Estimated-Failure-Time",
            ],
        ),
    )
    data.load()
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    transform = Standardize()
    learner = LightningLearner(
        module=MultilayerPerceptron(
            learning_rate=1e-3,
            input_size=3,
            output_size=1,
            hidden_dimensions=[10, 10],
        ),
        max_epochs=5,
    )
    inputs = ["y-Amplitude", "z-Amplitude", "Growth-rate"]
    outputs = ["Estimated-Failure-Time"]

    model = learn_offline(
        train,
        learner,
        inputs,
        outputs,
        input_transform=transform,
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
