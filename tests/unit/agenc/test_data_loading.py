import unittest
from pathlib import Path

import polars as pl
from agenc.core import Metadata


class TestMetadata(unittest.TestCase):
    def test_load_dataset_from_csv(self) -> None:
        # Create a temporary CSV file for testing
        data_file = "test_data.csv"
        data = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        data.write_csv(data_file)

        # Create Path objects for the file paths
        data_file_path = Path(data_file)

        metadata = Metadata(
            data_path=[data_file_path], test_data_path=[], features=[]
        )

        loaded_data = metadata.load_dataset()
        assert isinstance(loaded_data, pl.DataFrame)
        assert len(loaded_data) == 3
        data_file_path.unlink()

    def test_load_test_dataset_from_csv(self) -> None:
        # Create temporary CSV files for testing
        test_data_file = "test_test_data.csv"
        data = pl.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
        data.write_csv(test_data_file)

        # Create Path objects for the file paths
        test_data_file_path = Path(test_data_file)

        metadata = Metadata(
            data_path=[], test_data_path=[test_data_file_path], features=[]
        )

        loaded_data = metadata.load_test_dataset()
        assert isinstance(loaded_data, pl.DataFrame)
        assert len(loaded_data) == 3
        test_data_file_path.unlink()


if __name__ == "__main__":
    unittest.main()
