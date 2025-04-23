from __future__ import annotations

import math
from collections.abc import Collection, Iterable
from itertools import islice
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import polars as pl
from ruamel.yaml import YAML
from tqdm import tqdm
from typing_extensions import Self, override

from flowcean.core import OfflineEnvironment
from flowcean.core.environment.incremental import IncrementalEnvironment


class DataFrame(OfflineEnvironment):
    """A dataset environment.

    This environment represents static tabular datasets.

    Attributes:
        data: The data to represent.
    """

    data: pl.LazyFrame
    _length: int | None = None

    def __init__(self, data: pl.DataFrame | pl.LazyFrame) -> None:
        """Initialize the dataset environment.

        Args:
            data: The data to represent.
        """
        if isinstance(data, pl.DataFrame):
            self.data = data.lazy()
            self._length = len(data)
        else:
            self.data = data
        super().__init__()

    def to_incremental(self, batch_size: int = 1) -> IncrementalDataFrame:
        """Convert the DataFrame to an incremental environment.

        Args:
            batch_size: The size of each batch. Defaults to 1.
        """
        return IncrementalDataFrame(self, batch_size)

    @classmethod
    def from_csv(cls, path: str | Path, separator: str = ",") -> Self:
        """Load a dataset from a CSV file.

        Args:
            path: Path to the CSV file.
            separator: Value separator. Defaults to ",".
        """
        data = pl.scan_csv(path, separator=separator)
        data = data.rename(lambda column_name: column_name.strip())
        return cls(data)

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """Load a dataset from a JSON file.

        Args:
            path: Path to the JSON file.
        """
        data = pl.read_json(path)
        return cls(data)

    @classmethod
    def from_parquet(cls, path: str | Path) -> Self:
        """Load a dataset from a Parquet file.

        Args:
            path: Path to the Parquet file.
        """
        data = pl.scan_parquet(path)
        return cls(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a dataset from a YAML file.

        Args:
            path: Path to the YAML file.
        """
        data = pl.LazyFrame(YAML(typ="safe").load(path))
        return cls(data)

    @classmethod
    def from_uri(cls, uri: str) -> Self:
        """Load a dataset from a URI.

        Args:
            uri: The URI to load the dataset from.
        """
        path = _file_uri_to_path(uri)
        suffix = path.suffix

        if suffix == ".csv":
            return cls.from_csv(path)
        if suffix == ".json":
            return cls.from_json(path)
        if suffix == ".parquet":
            return cls.from_parquet(path)
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)

        raise UnsupportedFileTypeError(suffix)

    @override
    def _observe(self) -> pl.LazyFrame:
        return self.data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._length is None:
            # This operation is potentially very slow / costly
            self._length = cast(
                "int",
                self.data.select(pl.len()).collect().item(),
            )
        return self._length


class IncrementalDataFrame(IncrementalEnvironment):
    """Wraps a DataFrame in an incremental environment."""

    index: int = 0
    current_observation: pl.LazyFrame
    data: DataFrame
    batch_size: int

    def __init__(self, data: DataFrame, batch_size: int) -> None:
        """Create an incremental environment from a DataFrame.

        Args:
            data: The DataFrame to wrap.
            batch_size: The size of each batch.
        """
        super().__init__()

        self.transform = data.transform
        self.data = data
        self.batch_size = batch_size
        self.current_observation = data.data.slice(0, batch_size)

    def num_steps(self) -> int | None:
        return math.ceil(len(self.data) / self.batch_size)

    def step(self) -> None:
        """Advance the environment by one step."""
        if self.index >= len(self.data):
            raise StopIteration

        self.current_observation = self.data.data.slice(
            self.index,
            self.batch_size,
        )
        self.index += self.batch_size

    @override
    def _observe(self) -> pl.LazyFrame:
        """Return the current raw observation.

        If any transforms were attached to the original environment, they will
        be applied to the current observation before returning it!
        """
        return self.current_observation


def _file_uri_to_path(uri: str) -> Path:
    url = urlparse(uri)
    if url.scheme != "file":
        raise InvalidUriSchemeError(url.scheme)
    return Path(url.path)


class InvalidUriSchemeError(Exception):
    """Exception raised when an URI scheme is invalid."""

    def __init__(self, scheme: str) -> None:
        """Initialize the InvalidUriSchemeError.

        Args:
            scheme: Invalid URI scheme.
        """
        super().__init__(
            f"only file URIs can be converted to a path, but got `{scheme}`",
        )


class UnsupportedFileTypeError(Exception):
    """Exception raised when a file type is not supported."""

    def __init__(self, suffix: str) -> None:
        """Initialize the UnsupportedFileTypeError.

        Args:
            suffix: File type suffix.
        """
        super().__init__(f"file type `{suffix}` is not supported")


def collect(
    environment: Iterable[pl.LazyFrame] | Collection[pl.LazyFrame],
    n: int | None = None,
    *,
    progress_bar: bool | dict[str, Any] = True,
) -> DataFrame:
    """Collect data from an environment.

    Args:
        environment: The environment to collect data from.
        n: Number of samples to collect. If None, all samples are collected.
        progress_bar: Whether to show a progress bar. If a dictionary is
            provided, it will be passed to the progress bar.

    Returns:
        The collected dataset.
    """
    samples = islice(environment, n)

    if n is not None:
        total = n
    elif isinstance(environment, Collection):
        total = len(environment)
    else:
        total = None

    if isinstance(progress_bar, dict):
        progress_bar.setdefault("desc", "Collecting samples")
        progress_bar.setdefault("total", total)
        samples = tqdm(
            samples,
            **progress_bar,
        )
    elif progress_bar:
        samples = tqdm(samples, desc="Collecting samples", total=total)

    data = pl.concat(samples, how="vertical")
    return DataFrame(data)
