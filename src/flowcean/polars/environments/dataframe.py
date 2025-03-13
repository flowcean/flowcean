from __future__ import annotations

import hashlib
from collections.abc import Collection, Generator, Iterable
from itertools import islice
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import cloudpickle
import polars as pl
from ruamel.yaml import YAML
from tqdm import tqdm
from typing_extensions import Self, override

from flowcean.core import OfflineEnvironment


class DataFrame(OfflineEnvironment):
    """A dataset environment.

    This environment represents static tabular datasets.

    Attributes:
        data: The data to represent.
    """

    data: pl.LazyFrame
    _hash: bytes | None = None
    _length: int | None = None

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        data_hash: bytes | None = None,
    ) -> None:
        """Initialize the dataset environment.

        Args:
            data: The data to represent.
            data_hash: The hash of the data. If None, it will be computed from
                the dataframe which is potentially slow and expensive.
        """
        if isinstance(data, pl.DataFrame):
            self.data = data.lazy()
            self._length = len(data)
        else:
            self.data = data

        self._hash = data_hash
        super().__init__()

    @classmethod
    def from_csv(cls, path: str | Path, separator: str = ",") -> Self:
        """Load a dataset from a CSV file.

        Args:
            path: Path to the CSV file.
            separator: Value separator. Defaults to ",".
        """
        data = pl.scan_csv(path, separator=separator)
        data = data.rename(lambda column_name: column_name.strip())
        return cls(data, data_hash=_hash_from_path(path))

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """Load a dataset from a JSON file.

        Args:
            path: Path to the JSON file.
        """
        data = pl.read_json(path)
        return cls(data, data_hash=_hash_from_path(path))

    @classmethod
    def from_parquet(cls, path: str | Path) -> Self:
        """Load a dataset from a Parquet file.

        Args:
            path: Path to the Parquet file.
        """
        data = pl.scan_parquet(path)
        return cls(data, data_hash=_hash_from_path(path))

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a dataset from a YAML file.

        Args:
            path: Path to the YAML file.
        """
        data = pl.LazyFrame(YAML(typ="safe").load(path))
        return cls(data, data_hash=_hash_from_path(path))

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

    @override
    def hash(self) -> bytes:
        if self._hash is None:
            self._hash = _hash_from_dataframe(self.data)
        return self._hash

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._length is None:
            # This operation is potentially very slow / costly
            self._length = cast(
                int,
                self.data.select(pl.len()).collect().item(),
            )
        return self._length


def _file_uri_to_path(uri: str) -> Path:
    url = urlparse(uri)
    if url.scheme != "file":
        raise InvalidUriSchemeError(url.scheme)
    return Path(url.path)


def _hash_from_dataframe(data: pl.LazyFrame) -> bytes:
    hasher = hashlib.sha256()
    schema = data.collect_schema()
    hasher.update(cloudpickle.dumps(schema))

    for batch in _iter_slices(data, 42):
        # This is a workaround for the fact that Polars does not support
        # hashing DataFrames directly
        hasher.update(batch.to_numpy().dumps())
    return hasher.digest()


def _hash_from_path(path: str | Path) -> bytes:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)
    return hasher.digest()


def _iter_slices(
    df: pl.LazyFrame,
    batch_size: int,
) -> Generator[pl.DataFrame, Any, None]:
    def get_batch(
        df: pl.LazyFrame,
        offset: int,
        batch_size: int,
    ) -> pl.DataFrame:
        return df.slice(offset, batch_size).collect(streaming=True)

    batch = get_batch(df, 0, batch_size)
    # Yield once even if we got passed an empty LazyFrame
    yield batch
    offset = len(batch)
    if offset:
        while True:
            batch = get_batch(df, offset, batch_size)
            len_ = len(batch)
            if len_:
                offset += len_
                yield batch
            else:
                break


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
