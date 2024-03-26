import logging

import flowcean.cli
from flowcean.data.train_test_split import TrainTestSplit
from flowcean.data.uri import UriDataLoader
from flowcean.learners.lightning import LightningLearner, MultilayerPerceptron
from flowcean.metrics import MeanAbsoluteError, MeanSquaredError, evaluate
from flowcean.strategies.offline import learn_offline
from flowcean.transforms import Select, Standardize

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()

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
