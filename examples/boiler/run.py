from pathlib import Path

import polars as pl

from agenc.data.metadata import Feature, Metadata
from agenc.data.split import train_test_split
from agenc.metrics.regression import MeanAbsoluteError
from agenc.regression_tree import RegressionTree
from agenc.transforms import Select, SlidingWindow, StandardScaler


def main():
    metadata = Metadata(
        data_path=Path("data/trace_287401a5.csv"), features=[Feature(name="x")]
    )
    dataset: pl.DataFrame = metadata.load_dataset()

    dataset = Select(features=["reference", "temperature"])(dataset)
    dataset = StandardScaler()(dataset)
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
    print(f"{metric.name}: {result}")


if __name__ == "__main__":
    main()
