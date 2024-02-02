import polars as pl
from agenc.data.split import TrainTestSplit
from agenc.data.uri import UriDataLoader
from agenc.learners.regression_tree import RegressionTree
from agenc.metrics.regression import MeanAbsoluteError
from agenc.transforms import Select, SlidingWindow, Standardize


def main() -> None:
    dataset: pl.DataFrame = UriDataLoader(
        uri="file:./data/trace_287401a5.csv",
    ).load()

    train_data, test_data = TrainTestSplit(
        ratio=0.8,
        shuffle=False,
    )(dataset)

    transforms = (
        Select(features=["reference", "temperature"])
        | Standardize()
        | SlidingWindow(window_size=3)
    )

    transforms.fit(train_data)

    train_data = transforms(train_data)
    test_data = transforms(test_data)

    learner = RegressionTree()
    inputs = [
        "reference_0",
        "temperature_0",
        "reference_1",
        "temperature_1",
        "reference_2",
    ]
    outputs = ["temperature_2"]
    learner.train(
        input_features=train_data.select(inputs),
        output_features=train_data.select(outputs),
    )

    predictions = learner.predict(input_features=test_data.select(inputs))

    metric = MeanAbsoluteError()
    result = metric(
        test_data.select(outputs).to_numpy(),
        predictions,
    )
    print(f"{metric.__class__.__name__}: {result}")


if __name__ == "__main__":
    main()
