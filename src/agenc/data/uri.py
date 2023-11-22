from pathlib import Path
from urllib.parse import urlparse

import polars as pl
from typing_extensions import override

from agenc.core import DataLoader
from agenc.data.csv import CsvDataLoader
from agenc.data.parquet import ParquetDataLoader


class UriDataLoader(DataLoader):
    """DataLoader for files specified by an URI."""

    def __init__(self, uri: str):
        """Initialize the UriDataLoader.

        Args:
            uri: Path to the URI file.
        """
        self.uri = uri

    @override
    def load(self) -> pl.DataFrame:
        path = _file_uri_to_path(self.uri, Path.cwd())
        suffix = path.suffix
        if suffix == ".csv":
            return CsvDataLoader(path).load()
        if suffix == ".parquet":
            return ParquetDataLoader(path).load()

        supported_file_types = [".csv", ".parquet"]
        raise ValueError(
            "file type of data source has to be one of"
            f" {supported_file_types}, but got: `{suffix}`"
        )


def _file_uri_to_path(uri: str, root: Path) -> Path:
    url = urlparse(uri)
    if url.scheme != "file":
        raise ValueError(
            "only local files are supported as data source, but got:"
            f" `{url.scheme}`",
        )
    data_source = Path(url.path)
    if not data_source.is_absolute():
        data_source = (root / data_source).absolute()
    return data_source
