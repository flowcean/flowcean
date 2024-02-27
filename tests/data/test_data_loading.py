import tempfile
import unittest
from pathlib import Path

import polars as pl
import pytest
from agenc.data import CsvDataLoader, ParquetDataLoader
from agenc.data.uri import _file_uri_to_path
from polars.testing import assert_frame_equal


class TestDataloading(unittest.TestCase):
    def test_csv_loader(self) -> None:
        data = pl.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        })

        with tempfile.NamedTemporaryFile() as f:
            data.write_csv(f.name)
            datapath = Path(f.name)
            dataloader = CsvDataLoader(path=datapath)
            loaded_data = dataloader.load()
            assert_frame_equal(loaded_data, data)

    def test_parquet_loader(self) -> None:
        data = pl.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        })

        with tempfile.NamedTemporaryFile() as f:
            data.write_parquet(f.name)
            datapath = Path(f.name)
            dataloader = ParquetDataLoader(path=datapath)
            loaded_data = dataloader.load()
            assert_frame_equal(loaded_data, data)


class TestFileUriToPath(unittest.TestCase):
    def test_file_uri_to_path_absolute_path(self) -> None:
        root_path = Path("/root")
        uri = "file:/absolute/path/to/data.csv"
        result = _file_uri_to_path(uri, root_path)
        assert result == Path("/absolute/path/to/data.csv")

    def test_file_uri_to_path_relative_path(self) -> None:
        root_path = Path("/root")
        uri = "file:relative/path/to/data.csv"
        result = _file_uri_to_path(uri, root_path)
        assert result == Path("/root/relative/path/to/data.csv")

    def test_non_file_uri(self) -> None:
        root_path = Path("/root")
        uri = "http://example.com/data.csv"
        with pytest.raises(
            ValueError,
            match="only file URIs can be converted to a path, but got `http`",
        ):
            _file_uri_to_path(uri, root_path)


if __name__ == "__main__":
    unittest.main()
