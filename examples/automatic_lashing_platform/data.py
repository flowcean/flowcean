from pathlib import Path

import flowcean.utils.random
from flowcean.polars import (
    DataFrame,
    DiscreteDerivative,
    Filter,
    Flatten,
    Resample,
    Select,
    TimeWindow,
    TrainTestSplit,
)

EXPERIMENT_DATA_PATH: str = "./data/alp_sim_data.parquet"


def main() -> None:
    flowcean.utils.random.initialize_random(42)

    data_path = Path(EXPERIMENT_DATA_PATH)
    data = (
        DataFrame.from_parquet(data_path)
        | Select(
            [
                "p_accumulator",
                "container_weight",
                "active_valve_count",
                "T",
            ],
        )
        | Filter("active_valve_count > 0")
        | Resample(0.25)
        | TimeWindow(
            time_start=0,
            time_end=6,
        )
        | DiscreteDerivative("p_accumulator")
        | Flatten()
    )
    data.observe().sink_parquet(
        data_path.with_name(f"{data_path.stem}.processed.parquet"),
    )
    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=True).split(data)
    train_env.observe().sink_parquet(
        data_path.with_name(f"{data_path.stem}.train.parquet"),
    )
    test_env.observe().sink_parquet(
        data_path.with_name(f"{data_path.stem}.test.parquet"),
    )


if __name__ == "__main__":
    main()
