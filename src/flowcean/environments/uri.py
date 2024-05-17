from pathlib import Path
from typing import Self, override
from urllib.parse import urlparse

import polars as pl

from flowcean.core import OfflineEnvironment
from flowcean.core.environment import NotLoadedError
from flowcean.environments.csv import CsvDataLoader
from flowcean.environments.parquet import ParquetDataLoader


class UnsupportedFileTypeError(Exception):
    def __init__(self, suffix: str) -> None:
        super().__init__(f"file type `{suffix}` is not supported")


class UriDataLoader(OfflineEnvironment):
    """DataLoader for files specified by an URI."""

    uri: str
    data_loader: OfflineEnvironment | None = None

    def __init__(self, uri: str) -> None:
        """Initialize the UriDataLoader.

        Args:
            uri: Path to the URI file.
        """
        self.uri = uri

    @override
    def load(self) -> Self:
        path = _file_uri_to_path(self.uri)
        suffix = path.suffix
        if suffix == ".csv":
            self.data_loader = CsvDataLoader(path)
        elif suffix == ".parquet":
            self.data_loader = ParquetDataLoader(path)
        else:
            raise UnsupportedFileTypeError(suffix)
        self.data_loader.load()
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        if self.data_loader is None:
            raise NotLoadedError
        return self.data_loader.get_data()


class InvalidUriSchemeError(Exception):
    def __init__(self, scheme: str) -> None:
        super().__init__(
            f"only file URIs can be converted to a path, but got `{scheme}`",
        )


def _file_uri_to_path(uri: str) -> Path:
    url = urlparse(uri)
    if url.scheme != "file":
        raise InvalidUriSchemeError(url.scheme)
    return Path(url.path)
