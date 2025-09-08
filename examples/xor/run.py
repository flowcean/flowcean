from pathlib import Path

import polars as pl

from flowcean.core.model import Model
from flowcean.core.strategies import learn_offline
from flowcean.core.tool import start_prediction_loop, test_model
from flowcean.core.tool.testing.domain import Discrete
from flowcean.core.tool.testing.generator import CombinationGenerator
from flowcean.core.tool.testing.predicates import PolarsPredicate
from flowcean.polars.adapter.dataframe_adapter import DataFrameAdapter
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.sklearn.regression_tree import RegressionTree


def main() -> None:
    # Learn a model for the XOR function

    # Load the data from a CSV file
    data = DataFrame.from_csv("./data/data.csv")

    # Create a regression tree model and train it on the data
    learner = RegressionTree()
    inputs = [
        "x",
        "y",
    ]
    outputs = ["z"]

    model = learn_offline(
        data,
        learner,
        inputs,
        outputs,
    )

    # Save the trained model to a file for demonstration purposes
    model.save(Path("xor_model.fml"))

    # Use the trained model to make predictions

    model = Model.load(Path("xor_model.fml"))
    # Create a fake adapter from a DataFrame
    adapter = DataFrameAdapter(
        DataFrame.from_csv("./data/data.csv"),
        input_features=["x", "y"],
        result_path="result.csv",
    )

    # Run the prediction loop. The loop is blocking until the Adapter signals
    # a stop when nor more data is available or the program is interrupted.
    start_prediction_loop(
        model,
        adapter,
    )

    # Use the trained model for testing
    test_generator = CombinationGenerator(
        Discrete("x", [0, 1]),
        Discrete("y", [0, 1]),
    )

    test_generator.save_excel("xor_test_cases.xlsx")
    test_generator.reset()

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
