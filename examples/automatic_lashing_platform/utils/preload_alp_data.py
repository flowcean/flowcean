import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tqdm import tqdm

import flowcean.cli
from flowcean.core import ChainedOfflineEnvironments
from flowcean.polars import (
    DataFrame,
    JoinedOfflineEnvironment,
    ToTimeSeries,
)

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize()
    time_start = time.time()
    data = ChainedOfflineEnvironments(
        [
            JoinedOfflineEnvironment(
                (
                    DataFrame.from_parquet(path) | ToTimeSeries("t"),
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
