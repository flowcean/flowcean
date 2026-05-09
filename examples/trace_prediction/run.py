from dataclasses import dataclass
from pathlib import Path

import polars as pl

import flowcean.cli
from flowcean.core import evaluate_offline, learn_offline
from flowcean.polars import DataFrame, SlidingWindow, TrainTestSplit
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanSquaredError,
    RegressionTree,
)
from flowcean.utils import initialize_random

DATA_PATH = Path(__file__).parent / "data" / "thermostat_trace.csv"
INPUTS = [
    f"{column}_{step}"
    for step in range(3)
    for column in ("temperature", "target", "heating")
]
OUTPUTS = ["temperature_3"]


@dataclass
class TracePredictionData:
    train: DataFrame
    test: DataFrame
    source_rows: int
    train_rows: int
    test_rows: int
    input_features: list[str]
    output_features: list[str]


def _row_count(environment: DataFrame) -> int:
    return environment.observe().select(pl.len()).collect().item()


def load_trace_data(path: Path = DATA_PATH) -> TracePredictionData:
    environment = DataFrame.from_csv(path)
    (train, test) = TrainTestSplit(ratio=0.8, shuffle=False).split(environment)
    window = SlidingWindow(window_size=4)
    train_windows = train | window
    test_windows = test | window
    return TracePredictionData(
        train=train_windows,
        test=test_windows,
        source_rows=_row_count(environment),
        train_rows=_row_count(train_windows),
        test_rows=_row_count(test_windows),
        input_features=INPUTS,
        output_features=OUTPUTS,
    )


def format_data_summary(data: TracePredictionData) -> str:
    return "\n".join(
        [
            "Trace prediction example",
            f"source trace rows: {data.source_rows}",
            "transform: SlidingWindow(window_size=4)",
            f"training windows: {data.train_rows}",
            f"test windows: {data.test_rows}",
            f"inputs: {', '.join(data.input_features)}",
            f"outputs: {', '.join(data.output_features)}",
            "learner: RegressionTree(max_depth=5, random_state=42)",
        ],
    )


def train_and_evaluate() -> None:
    data = load_trace_data()
    print(format_data_summary(data))
    learner = RegressionTree(max_depth=5, random_state=42)
    model = learn_offline(
        data.train,
        learner,
        data.input_features,
        data.output_features,
    )
    report = evaluate_offline(
        model,
        data.test,
        data.input_features,
        data.output_features,
        [MeanAbsoluteError(), MeanSquaredError()],
    )

    test_frame = data.test.observe()
    predictions = model.predict(test_frame.select(data.input_features))
    preview = (
        pl.concat(
            (
                test_frame.select(data.output_features),
                predictions.rename({"temperature_3": "prediction"}),
            ),
            how="horizontal",
        )
        .head(5)
        .collect()
    )
    print("evaluation:")
    print(report)
    print("prediction preview:")
    print(preview)
    print(
        "interpretation: predicted temperatures should stay close to the "
        "held-out trace.",
    )


def main() -> None:
    flowcean.cli.initialize()
    initialize_random(42)
    train_and_evaluate()


if __name__ == "__main__":
    main()
