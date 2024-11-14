import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.environments.dataset import Dataset


class TestDataset(unittest.TestCase):
    def test_loading(self) -> None:
        df = pl.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            },
        )
        data = Dataset(df)
        assert_frame_equal(df, data.observe())


if __name__ == "__main__":
    unittest.main()
