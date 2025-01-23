import unittest
from datetime import UTC, datetime

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.transforms import SliceTimeSeries


class SliceTimeSeriesTransform(unittest.TestCase):
    def test_slicetimeseries(self) -> None:
        transform = SliceTimeSeries(num_slices=1, slice_length=1)

        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 1, 0, tzinfo=UTC, ),
                            "value": {"x": 1.2},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 2, 0, tzinfo=UTC, ),
                            "value": {"x": 2.4},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 3, 0, tzinfo=UTC, ),
                            "value": {"x":3.6},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 4, 0, tzinfo=UTC, ),
                            "value": {"x": 4.8},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 5, 0, tzinfo=UTC, ),
                            "value": {"x": 5.6},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 6, 0, tzinfo=UTC, ),
                            "value": {"x": 7.4},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 8, 0, tzinfo=UTC, ),
                            "value": {"x":8.1},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 9, 0, tzinfo=UTC, ),
                            "value": {"x": 8.5},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 10, 0, tzinfo=UTC, ),
                            "value": {"x": 9.2},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 11, 0, tzinfo=UTC, ),
                            "value": {"x": 10.0},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 12, 0, tzinfo=UTC, ),
                            "value": {"x": 11.3},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 13, 0, tzinfo=UTC, ),
                            "value": {"x": 12.6},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 14, 0, tzinfo=UTC, ),
                            "value": {"x": 13.9},
                        },
                    ],
                ],
                "feature_b": [
                    [   
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 0, 0, tzinfo=UTC, ),
                            "value": {"x": 1.0},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 5, 0, tzinfo=UTC, ),
                            "value": {"x": 2.0},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 10, 0, tzinfo=UTC, ),
                            "value": {"x": 3.0},
                        },
                        {
                            "time": datetime(2024, 6, 25, 12, 26, 15, 0, tzinfo=UTC, ),
                            "value": {"x": 4.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()
        exit()
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
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
