import agenc.cli
from agenc.data.train_test_split import TrainTestSplit
from agenc.learners.regression_tree import RegressionTree
from agenc.metrics import MeanAbsoluteError, MeanSquaredError, evaluate
from agenc.strategies.offline import learn_offline
from agenc.transforms.select import Select
from loader import AlpDataLoader


def main() -> None:
    agenc.cli.initialize()

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
