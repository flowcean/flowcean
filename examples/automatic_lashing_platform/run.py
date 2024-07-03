from loader import AlpDataLoader

import flowcean.cli
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.regression_tree import RegressionTree
from flowcean.metrics import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms.select import Select


def main() -> None:
    flowcean.cli.initialize_logging()

    data = AlpDataLoader(path="./data/").with_transform(
        Select(
            [
                "^p_cylinder1_.*$",
                "activeValveCount",
                "containerWeight",
            ],
        ),
    )
    data.load()
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    learner = RegressionTree()
    inputs = [
        "^p_cylinder1_.*$",
        "activeValveCount",
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
