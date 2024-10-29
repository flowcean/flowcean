import tempfile
import unittest
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.environments.parquet import ParquetDataLoader


class TestParquetDataLoader(unittest.TestCase):
    def test_parquet_loader(self) -> None:
        data = pl.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            },
        )

        with tempfile.NamedTemporaryFile() as f:
            data.write_parquet(f.name)
            datapath = Path(f.name)
            dataloader = ParquetDataLoader(path=datapath)
            loaded_data = dataloader.observe()
            assert_frame_equal(loaded_data, data)


if __name__ == "__main__":
    unittest.main()
