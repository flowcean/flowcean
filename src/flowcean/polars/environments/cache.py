import hashlib
import logging
from pathlib import Path

import polars as pl

from flowcean.core import (
    HashingNotSupportedError,
    Observable,
    TransformedObservable,
)
from flowcean.core.data import Data
from flowcean.polars import DataFrame

logger = logging.getLogger(__name__)


class Cache(Observable):
    """A cache environment."""

    def __init__(
        self,
        base_environment: TransformedObservable,
        *,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.base_environment = base_environment
        self.caching_supported = True

        # Calculate the hash of the base environment and it's transforms
        try:
            self.environment_hash = base_environment.hash()
        except HashingNotSupportedError:
            self.caching_supported = False
            logger.warning("Caching is not supported by the base environment.")
            return

        # Combine the environment hash with the transform hash and convert to
        # an hex string used as a filename for the cache
        hasher = hashlib.sha256()
        hasher.update(self.environment_hash)
        hasher.update(
            base_environment.transform.__hash__().to_bytes(
                16,
                "big",
                signed=True,
            ),
        )
        hash_value = hasher.digest()

        cache_dir_path = Path(
            cache_dir if cache_dir is not None else ".cache",
        )
        # Make sure the path exists
        cache_dir_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.cache_path = cache_dir_path.joinpath(
            f"{hash_value.hex()}.parquet",
        )

        if not self.cache_path.exists():
            self.cached_environment = None
            logger.info("Cache file %s does not exist.", self.cache_path)
            return

        self.cached_environment = DataFrame.from_parquet(
            self.cache_path,
        )

    def hash(self) -> bytes:
        if self.cached_environment is not None:
            return self.cached_environment.hash()
        return self.base_environment.hash()

    def observe(self) -> Data:
        if not self.caching_supported:
            return self.base_environment.observe()

        if self.cached_environment is None:
            # Caching *is* supported, but the cache file does not exist
            data = self.base_environment.observe()
            if isinstance(data, pl.DataFrame):
                data.write_parquet(self.cache_path)
            elif isinstance(data, pl.LazyFrame):
                data.sink_parquet(self.cache_path)
            else:
                msg = f"Unsupported data type {type(data)} for caching."
                raise ValueError(msg)

            # We directly load the data from the cache file and return it
            self.cached_environment = DataFrame.from_parquet(self.cache_path)

        return self.cached_environment.observe()
