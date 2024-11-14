import logging
import time
from pathlib import Path

from tqdm import tqdm

import flowcean.cli
from flowcean.core.environment.chained import ChainedOfflineEnvironments
from flowcean.environments.json import JsonDataLoader
from flowcean.environments.parquet import ParquetDataLoader
from flowcean.transforms import ToTimeSeries

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()
    time_start = time.time()
    data = ChainedOfflineEnvironments(
        [
            ParquetDataLoader(path)
            .with_transform(ToTimeSeries("t"))
            .join(JsonDataLoader(path.with_suffix(".json")))
            for path in tqdm(
                list(Path("./data").glob("*.parquet")),
                desc="Loading environments",
            )
        ]
    )
    time_end = time.time()
    logger.info("took %.5f s to load data", time_end - time_start)

    data.observe().write_parquet(Path("./alp_sim_data.parquet"))


if __name__ == "__main__":
    main()
