import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import SlidingWindow


class SlidingWindowTransform(unittest.TestCase):
    def test_sliding_window(self) -> None:
        transform = SlidingWindow(window_size=3)

        data_frame = pl.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [10, 20, 30, 40],
                "c": [100, 200, 300, 400],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a_0": [1, 2],
                    "b_0": [10, 20],
                    "c_0": [100, 200],
                    "a_1": [2, 3],
                    "b_1": [20, 30],
                    "c_1": [200, 300],
                    "a_2": [3, 4],
                    "b_2": [30, 40],
                    "c_2": [300, 400],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
