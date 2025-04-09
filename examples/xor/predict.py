from pathlib import Path

from flowcean.core.model import Model
from flowcean.core.tool.predict import start_prediction_loop
from flowcean.polars.adapter.dataframe_adapter import DataFrameAdapter
from flowcean.polars.environments.dataframe import DataFrame


def main() -> None:
    # Load the trained model
    model = Model.load(Path("xor_model.fml"))

    # Create a fake adapter from a DataFrame
    adapter = DataFrameAdapter(
        DataFrame.from_csv("data.csv"),
        input_features=["x", "y"],
        result_path="result.csv",
    )

    # Run the prediction loop. The loop is blocking until the Adapter signals
    # a stop when nor more data is available or the program is interrupted.
    start_prediction_loop(
        model,
        adapter,
    )


if __name__ == "__main__":
    main()
