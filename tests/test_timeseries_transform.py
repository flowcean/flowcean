import unittest

import polars as pl
from flowcean.transforms import Flatten
from polars.testing import assert_frame_equal


class TimeSeriesTransform(unittest.TestCase):
    def test_flatten(self) -> None:
        flatten_transform = Flatten()

        data_frame = pl.DataFrame(
            [
                {
                    "a": [0, 1],
                    "b": [0, 1, 2],
                    "c": 3,
                },
                {
                    "a": [1, 2],
                    "b": [3, 4, 5],
                    "c": 6,
                },
            ],
        )

        transformed_data = flatten_transform.transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"a_0": 0, "a_1": 1, "b_0": 0, "b_1": 1, "b_2": 2, "c": 3},
                    {"a_0": 1, "a_1": 2, "b_0": 3, "b_1": 4, "b_2": 5, "c": 6},
                ],
            ),
            check_column_order=False,
        )

    def test_flatten_selected(self) -> None:
        flatten_transform = Flatten(features=["a"])

        data_frame = pl.DataFrame(
            [
                {
                    "a": [0, 1],
                    "b": [0, 1, 2],
                    "c": 3,
                },
                {
                    "a": [1, 2],
                    "b": [3, 4, 5],
                    "c": 6,
                },
            ],
        )

        transformed_data = flatten_transform.transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"a_0": 0, "a_1": 1, "b": [0, 1, 2], "c": 3},
                    {"a_0": 1, "a_1": 2, "b": [3, 4, 5], "c": 6},
                ],
            ),
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
