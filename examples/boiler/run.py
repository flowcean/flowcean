from pathlib import Path

import polars as pl
from agenc.core import Feature, Metadata, train_test_split
from agenc.learners.regression_tree import RegressionTree
from agenc.metrics.regression import MeanAbsoluteError
from agenc.transforms import Select, SlidingWindow, Standardize


def main() -> None:
    metadata = Metadata(
        data_path=Path("data/trace_287401a5.csv"),
        features=[Feature(name="x")],
    )
    dataset: pl.DataFrame = metadata.load_dataset()

    dataset = Select(features=["reference", "temperature"])(dataset)
    dataset = Standardize(
        mean={
            "reference": 0.0,
            "temperature": 0.0,
        },
        std={
            "reference": 1.0,
            "temperature": 1.0,
        },
    )(dataset)
    dataset = SlidingWindow(window_size=3)(dataset)

    train_data, test_data = train_test_split(dataset, ratio=0.8)

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
        inputs=train_data.select(
            inputs,
        ).to_numpy(),
        outputs=train_data.select(outputs).to_numpy(),
    )

    predictions = learner.predict(inputs=test_data.select(inputs).to_numpy())

    metric = MeanAbsoluteError()
    result = metric(
        test_data.select(outputs).to_numpy(),
        predictions,
    )
    print(f"{metric.__class__.__name__}: {result}")


if __name__ == "__main__":
    main()
