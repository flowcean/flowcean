from pathlib import Path

import polars as pl

from flowcean.core.model import Model
from flowcean.core.tool import test_model
from flowcean.core.tool.testing.generator import (
    StochasticDataFrameGenerator,
)
from flowcean.core.tool.testing.generator.range import Uniform
from flowcean.core.tool.testing.predicates import NotPredicate, PolarsPredicate
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.polars.transforms.select import Select


def main() -> None:
    # Load the trained model
    model = Model.load(Path("xor_model.fml"))

    # Use the training data for the test
    test_data = (
        DataFrame.from_csv("data.csv")
        .with_transform(Select([pl.col("x"), pl.col("y")]))
        .to_incremental()
    )

    test_generator = StochasticDataFrameGenerator(
        [
            # Generate random values for x and y
            Uniform("x", 0, 1),
            Uniform("y", 0, 1),
        ],
        seed=42,
    )

    # Define a predicate to check with the data
    predicate = PolarsPredicate(
        pl.col("x").xor(pl.col("y")) == pl.col("z"),
    )

    # Test the model with the test data and predicate
    test_model(
        model,
        # test_data,
        test_generator,
        predicate,
        show_progress=True,
        stop_after=10,
    )


if __name__ == "__main__":
    main()
