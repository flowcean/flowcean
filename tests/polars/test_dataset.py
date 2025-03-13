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

    def test_hashing(self) -> None:
        df_a = DataFrame(
            pl.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                },
            ),
        )

        df_b = DataFrame(
            pl.DataFrame(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                },
            ),
        )

        assert df_a.hash() == df_b.hash()

    def test_hashing_ts(self) -> None:
        df = pl.DataFrame(
            {
                "scalar": [17],
                "time-series": [
                    [
                        {"time": 0, "value": 42},
                        {"time": 1, "value": 43},
                        {"time": 2, "value": 44},
                        {"time": 3, "value": 45},
                    ],
                ],
            },
        )

        df_a = DataFrame(df)
        df_b = DataFrame(df)

        assert df_a.hash() == df_b.hash()


if __name__ == "__main__":
    unittest.main()
