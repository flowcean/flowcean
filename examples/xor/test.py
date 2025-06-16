from pathlib import Path

import polars as pl

from flowcean.core.model import Model
from flowcean.core.tool import test_model
from flowcean.core.tool.testing.domain import Discrete
from flowcean.core.tool.testing.generator import CombinationGenerator
from flowcean.core.tool.testing.predicates import PolarsPredicate


def main() -> None:
    # Load the trained model
    model = Model.load(Path("xor_model.fml"))

    test_generator = CombinationGenerator(
        Discrete("x", [0, 1]),
        Discrete("y", [0, 1]),
    )

    # Define a predicate to check with the data
    predicate = PolarsPredicate(
        pl.col("x").xor(pl.col("y")) == pl.col("z"),
    )

    # Test the model with the test data and predicate
    test_model(
        model,
        test_generator,
        predicate,
        show_progress=True,
        stop_after=0,
    )


if __name__ == "__main__":
    main()
