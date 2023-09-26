import unittest

import polars as pl
from polars.testing import assert_frame_equal

from agenc.transforms import Select


class SelectTransform(unittest.TestCase):
    def test_select(self):
        transform = Select(features=["a", "c"])

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ]
        )
        transformed_data = transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"a": 1, "c": 3},
                    {"a": 4, "c": 6},
                    {"a": 7, "c": 9},
                    {"a": 10, "c": 12},
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
