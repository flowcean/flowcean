import unittest
from pathlib import Path

import pytest

from flowcean.polars import InvalidUriSchemeError
from flowcean.polars.environments.dataframe import _file_uri_to_path


class TestFileUriToPath(unittest.TestCase):
    def test_file_uri_to_path(self) -> None:
        uri = "file:relative/path/to/data.csv"
        result = _file_uri_to_path(uri)
        assert result == Path("relative/path/to/data.csv")

    def test_non_file_uri(self) -> None:
        uri = "http://example.com/data.csv"
        with pytest.raises(InvalidUriSchemeError):
            _file_uri_to_path(uri)


if __name__ == "__main__":
    unittest.main()
