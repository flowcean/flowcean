import logging

import agenc.cli
import polars as pl
from agenc.data.dataset import Dataset
from agenc.data.train_test_split import TrainTestSplit
from agenc.learners.linear_regression import LinearRegression
from agenc.metrics import MeanAbsoluteError, MeanSquaredError, evaluate
from agenc.strategies.online import learn_incremental

logger = logging.getLogger(__name__)

N = 1_000


def main() -> None:
    agenc.cli.initialize()

    data = Dataset(
        pl.DataFrame(
            {
                "x": pl.arange(0, N, eager=True).cast(pl.Float32) / N,
                "y": pl.arange(N, 0, -1, eager=True).cast(pl.Float32) / N,
            },
        ),
    )
    data.load()
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    learner = LinearRegression(
        input_size=1,
        output_size=1,
        learning_rate=0.01,
    )
    inputs = ["x"]
    outputs = ["y"]

    model = learn_incremental(
        train.as_stream(batch_size=1).load(),
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
