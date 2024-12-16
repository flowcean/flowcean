import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.transforms import Drop


class DropTransform(unittest.TestCase):
    def test_simple(self) -> None:
        transform = Drop(features=["a", "c"])

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        )
        transformed_data = transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"b": 2},
                    {"b": 5},
                    {"b": 8},
                    {"b": 11},
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
