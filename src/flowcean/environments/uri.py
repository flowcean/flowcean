from pathlib import Path
from urllib.parse import urlparse

from flowcean.environments.csv import CsvDataLoader
from flowcean.environments.dataset import Dataset
from flowcean.environments.parquet import ParquetDataLoader


class UnsupportedFileTypeError(Exception):
    """Exception raised when a file type is not supported."""

    def __init__(self, suffix: str) -> None:
        """Initialize the UnsupportedFileTypeError.

        Args:
            suffix: File type suffix.
        """
        super().__init__(f"file type `{suffix}` is not supported")


class UriDataLoader(Dataset):
    """DataLoader for files specified by an URI."""

    def __init__(self, uri: str) -> None:
        """Initialize the UriDataLoader.

        Args:
            uri: Path to the URI file.
        """
        path = _file_uri_to_path(uri)
        suffix = path.suffix
        if suffix == ".csv":
            data_loader = CsvDataLoader(path)
        elif suffix == ".parquet":
            data_loader = ParquetDataLoader(path)
        else:
            raise UnsupportedFileTypeError(suffix)
        super().__init__(data_loader.data)


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


def _file_uri_to_path(uri: str) -> Path:
    url = urlparse(uri)
    if url.scheme != "file":
        raise InvalidUriSchemeError(url.scheme)
    return Path(url.path)
