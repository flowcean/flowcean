import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import Median


class MedianTransform(unittest.TestCase):
    def test_single_feature(self) -> None:
        transform = Median("a")

        data_frame = pl.DataFrame(
            {
                "a": [
                    [
                        {"time": 0, "value": 0},
                        {"time": 1, "value": 5},
                        {"time": 2, "value": 10},
                    ],
                    [
                        {"time": 0, "value": -100},
                        {"time": 1, "value": 50},
                        {"time": 2, "value": 50},
                    ],
                ],
            },
        )

        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.concat(
                [
                    data_frame,
                    pl.DataFrame(
                        {
                            "a_median": [5.0, 50.0],
                        },
                    ),
                ],
                how="horizontal",
            ),
        )

    def test_replace_single_feature(self) -> None:
        transform = Median("a", replace=True)

        data_frame = pl.DataFrame(
            {
                "a": [
                    [
                        {"time": 0, "value": 0},
                        {"time": 1, "value": 5},
                        {"time": 2, "value": 10},
                    ],
                    [
                        {"time": 0, "value": -100},
                        {"time": 1, "value": 50},
                        {"time": 2, "value": 50},
                    ],
                ],
            },
        )

        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a": [5.0, 50.0],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
