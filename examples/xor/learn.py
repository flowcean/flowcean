from pathlib import Path

from flowcean.core.strategies import learn_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.sklearn.regression_tree import RegressionTree


def main() -> None:
    # Load the data from a CSV file
    data = DataFrame.from_csv("data.csv")

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

    # Save the trained model to a file
    model.save(Path("xor_model.fml"))


if __name__ == "__main__":
    main()
