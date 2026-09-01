import logging
from pathlib import Path

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

from .preload_alp_data import preload_alp_data

logger = logging.getLogger(__name__)

EXPERIMENT_DATA_PATH: str = "./data/alp_sim_data.parquet"


def split_dataset(
    path: Path | str = EXPERIMENT_DATA_PATH,
    seed: int = 42,
) -> None:
    data_path = Path(path)
    if not data_path.exists():
        logger.info("Processed data not found, preloading...")
        preload_alp_data()

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
    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=True).split(
        data,
        seed=seed,
    )
    train_env.observe().sink_parquet(
        data_path.with_name(f"{data_path.stem}.train.parquet"),
    )
    test_env.observe().sink_parquet(
        data_path.with_name(f"{data_path.stem}.test.parquet"),
    )


if __name__ == "__main__":
    split_dataset()
