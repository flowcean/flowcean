import unittest

import numpy as np
import polars as pl
from flowcean.transforms import Flatten
from polars.testing import assert_frame_equal


class TimeSeriesTransform(unittest.TestCase):
    def test_flatten(self) -> None:
        flatten_transform = Flatten()

        data_frame = pl.DataFrame(
            [
                {
                    "a": np.array([[0, 0], [1, 1]]),
                    "b": np.array([[0, 0], [1, 1], [2, 2]]),
                    "c": 3,
                },
                {
                    "a": np.array([[0, 1], [1, 2]]),
                    "b": np.array([[0, 3], [1, 4], [2, 5]]),
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


if __name__ == "__main__":
    unittest.main()
