import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import TimeWindow


class TimeWindowTransform(unittest.TestCase):
    def test_simple(self) -> None:
        transform = TimeWindow(time_start=1.0, time_end=2.0)
        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": 0,
                            "value": 0,
                        },
                        {
                            "time": 1,
                            "value": 1,
                        },
                        {
                            "time": 2,
                            "value": 2,
                        },
                        {
                            "time": 3,
                            "value": 3,
                        },
                    ],
                    [
                        {
                            "time": 0,
                            "value": 0,
                        },
                        {
                            "time": 2,
                            "value": 2,
                        },
                        {
                            "time": 5,
                            "value": 5,
                        },
                    ],
                ],
                "scalar": [1, 2],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "feature_a": [
                        [
                            {
                                "time": 1,
                                "value": 1,
                            },
                            {
                                "time": 2,
                                "value": 2,
                            },
                        ],
                        [
                            {
                                "time": 2,
                                "value": 2,
                            },
                        ],
                    ],
                    "scalar": [1, 2],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
