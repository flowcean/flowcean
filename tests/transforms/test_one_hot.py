import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.transforms import OneHot


class OneHotTransform(unittest.TestCase):
    def test_single(self) -> None:
        transform = OneHot(["a"])

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        )
        transform.fit(data_frame)
        transformed_data = transform.transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a_1": [1, 0, 0, 0],
                    "a_4": [0, 1, 0, 0],
                    "a_7": [0, 0, 1, 0],
                    "a_10": [0, 0, 0, 1],
                    "b": [2, 5, 8, 11],
                    "c": [3, 6, 9, 12],
                },
            ),
            check_column_order=False,
        )

    def test_multiple(self) -> None:
        transform = OneHot(["a", "b"])

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        )
        transform.fit(data_frame)
        transformed_data = transform.transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a_1": [1, 0, 0, 0],
                    "a_4": [0, 1, 0, 0],
                    "a_7": [0, 0, 1, 0],
                    "a_10": [0, 0, 0, 1],
                    "b_2": [1, 0, 0, 0],
                    "b_5": [0, 1, 0, 0],
                    "b_8": [0, 0, 1, 0],
                    "b_11": [0, 0, 0, 1],
                    "c": [3, 6, 9, 12],
                },
            ),
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
