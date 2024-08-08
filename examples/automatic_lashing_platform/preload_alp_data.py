import logging
import time
from pathlib import Path

from tqdm import tqdm

import flowcean.cli
from flowcean.core.environment import ChainEnvironment
from flowcean.environments.json import JsonDataLoader
from flowcean.environments.parquet import ParquetDataLoader

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()
    time_start = time.time()
    data = ChainEnvironment(
        *[
            ParquetDataLoader(path)
            .load()
            .to_time_series("t")
            .join(JsonDataLoader(path.with_suffix(".json")).load())
            for path in tqdm(
                list(Path("./data").glob("*.parquet")),
                desc="Loading environments",
            )
        ]
    )
    data.load()
    time_end = time.time()
    logger.info("took %.5f s to load data", time_end - time_start)

    data.get_data().write_parquet(Path("./alp_sim_data.parquet"))


if __name__ == "__main__":
    main()
