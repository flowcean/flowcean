import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import Unnest


class UnnestTransform(unittest.TestCase):
    def test_simple(self) -> None:
        transform = Unnest(features=["c"])

        data_frame = pl.Series(
            "c",
            [
                {"a": 1, "t": 1},
                {"a": 4, "t": 2},
                {"a": 7, "t": 3},
                {"a": 10, "t": 4},
                {"a": 15, "t": 5},
            ],
        ).to_frame()
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a": [1, 4, 7, 10, 15],
                    "t": [1, 2, 3, 4, 5],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
