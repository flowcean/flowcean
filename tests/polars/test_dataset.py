import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import DataFrame


class TestDataset(unittest.TestCase):
    def test_loading(self) -> None:
        df = pl.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            },
        )
        data = DataFrame(df)
        assert_frame_equal(df, data.observe().collect())


if __name__ == "__main__":
    unittest.main()
