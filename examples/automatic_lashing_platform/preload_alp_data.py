import logging
import time
from pathlib import Path

from tqdm import tqdm

import flowcean.cli
from flowcean.core import ChainedOfflineEnvironments
from flowcean.polars import (
    DataFrame,
    JoinedOfflineEnvironment,
    ToTimeSeries,
)

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()
    time_start = time.time()
    data = ChainedOfflineEnvironments(
        [
            JoinedOfflineEnvironment(
                (
                    DataFrame.from_parquet(path).with_transform(
                        ToTimeSeries("t"),
                    ),
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

    data.observe().collect(streaming=True).write_parquet(
        Path("./alp_sim_data.parquet"),
    )


if __name__ == "__main__":
    main()
