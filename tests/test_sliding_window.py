import unittest

import polars as pl
from agenc.transforms import SlidingWindow
from polars.testing import assert_frame_equal


class SlidingWindowTransform(unittest.TestCase):
    def test_sliding_window(self) -> None:
        transform = SlidingWindow(window_size=3)

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
                    {
                        "a_0": 1,
                        "a_1": 4,
                        "a_2": 7,
                        "b_0": 2,
                        "b_1": 5,
                        "b_2": 8,
                        "c_0": 3,
                        "c_1": 6,
                        "c_2": 9,
                    },
                    {
                        "a_0": 4,
                        "a_1": 7,
                        "a_2": 10,
                        "b_0": 5,
                        "b_1": 8,
                        "b_2": 11,
                        "c_0": 6,
                        "c_1": 9,
                        "c_2": 12,
                    },
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
