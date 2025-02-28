import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import DataFrame, ToTimeSeries


class TestToTimeSeries(unittest.TestCase):
    def test_to_time_series_single_time_feature(self) -> None:
        dataset = DataFrame(
            pl.DataFrame(
                {
                    "time_feature": [0, 1, 2, 3],
                    "feature_a": [42, 43, 44, 45],
                    "feature_b": [3, 2, 1, 0],
                },
            ).lazy(),
        )

        time_series_dataset = dataset.with_transform(
            ToTimeSeries("time_feature"),
        )

        assert_frame_equal(
            time_series_dataset.observe().collect(),
            pl.DataFrame(
                {
                    "feature_a": [
                        [
                            {"time": 0, "value": 42},
                            {"time": 1, "value": 43},
                            {"time": 2, "value": 44},
                            {"time": 3, "value": 45},
                        ],
                    ],
                    "feature_b": [
                        [
                            {"time": 0, "value": 3},
                            {"time": 1, "value": 2},
                            {"time": 2, "value": 1},
                            {"time": 3, "value": 0},
                        ],
                    ],
                },
            ),
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
