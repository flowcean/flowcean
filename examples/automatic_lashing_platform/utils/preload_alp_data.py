import logging
import time
from pathlib import Path
from typing import cast

import polars as pl
from tqdm import tqdm

import flowcean.cli
from flowcean.core import ChainedOfflineEnvironments
from flowcean.polars import (
    DataFrame,
    JoinedOfflineEnvironment,
    Lambda,
    ToTimeSeries,
)

logger = logging.getLogger(__name__)


def split_sensor_series(data: pl.LazyFrame) -> pl.LazyFrame:
    """Preserve the ALP dataset's separate scalar series for each sensor."""
    return pl.concat(
        [
            ToTimeSeries("t", name=feature)(data.select("t", feature))
            for feature in data.collect_schema().names()
            if feature != "t"
        ],
        how="horizontal_extend",
    )


def main() -> None:
    flowcean.cli.initialize()
    time_start = time.time()
    data = ChainedOfflineEnvironments(
        [
            JoinedOfflineEnvironment(
                (
                    DataFrame.from_parquet(path) | Lambda(split_sensor_series),
                    DataFrame.from_json(path.with_suffix(".json")),
                ),
            )
            for path in tqdm(
                list(Path("./data").glob("*.parquet")),
                desc="Loading environments",
            )
        ],
    )
    time_end = time.time()
    logger.info("took %.5f s to load data", time_end - time_start)

    cast("pl.LazyFrame", data.observe()).collect(
        engine="streaming",
    ).write_parquet(
        Path("./alp_sim_data.parquet"),
    )


if __name__ == "__main__":
    main()
