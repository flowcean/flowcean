import tempfile
import unittest
from pathlib import Path

import polars as pl
from agenc.data import CsvDataLoader


class TestDataloading(unittest.TestCase):
    def test_csvdataloader(self) -> None:
        data = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        with tempfile.NamedTemporaryFile() as f:
            data.write_csv(f.name)
            datapath = Path(f.name)
            dataloader = CsvDataLoader(path=datapath)
            loaded_data = dataloader.load()
            assert isinstance(loaded_data, pl.DataFrame)
            assert len(loaded_data) == 3


if __name__ == "__main__":
    unittest.main()
