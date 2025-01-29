import unittest
from datetime import UTC, datetime

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.transforms import SliceTimeSeries


class SliceTimeSeriesTransform(unittest.TestCase):
    def test_slicetimeseries(self) -> None:
        duration = 2
        input_data = pl.LazyFrame(
            {
                "feature_a":
                [
                    [
                        {"time": datetime(2024, 6, 25, 12, 26, 1, 0, tzinfo=UTC), "value": {"x": 1}},
                        {"time": datetime(2024, 6, 25, 12, 26, 2, 0, tzinfo=UTC), "value": {"x": 2}},
                        {"time": datetime(2024, 6, 25, 12, 26, 3, 0, tzinfo=UTC), "value": {"x": 3}},
                        {"time": datetime(2024, 6, 25, 12, 26, 4, 0, tzinfo=UTC), "value": {"x": 4}},
                        {"time": datetime(2024, 6, 25, 12, 26, 5, 0, tzinfo=UTC), "value": {"x": 5}},
                        {"time": datetime(2024, 6, 25, 12, 26, 6, 0, tzinfo=UTC), "value": {"x": 6}},
                        {"time": datetime(2024, 6, 25, 12, 26, 7, 0, tzinfo=UTC), "value": {"x": 7}},
                        {"time": datetime(2024, 6, 25, 12, 26, 8, 0, tzinfo=UTC), "value": {"x": 8}},
                        {"time": datetime(2024, 6, 25, 12, 26, 9, 0, tzinfo=UTC), "value": {"x": 9}},
                        {"time": datetime(2024, 6, 25, 12, 26, 10, 0, tzinfo=UTC), "value": {"x": 10}},
                        {"time": datetime(2024, 6, 25, 12, 26, 11, 0, tzinfo=UTC), "value": {"x": 11}},
                        {"time": datetime(2024, 6, 25, 12, 26, 12, 0, tzinfo=UTC), "value": {"x": 12}},
                        {"time": datetime(2024, 6, 25, 12, 26, 13, 0, tzinfo=UTC), "value": {"x": 13}},
                        {"time": datetime(2024, 6, 25, 12, 26, 14, 0, tzinfo=UTC), "value": {"x": 14}},
                    ],
                ],

                "counter":
                [
                    [   
                        {"time": datetime(2024, 6, 25, 12, 26, 5, 0, tzinfo=UTC), "value": {"x": 1.0}},
                        {"time": datetime(2024, 6, 25, 12, 26, 10, 0, tzinfo=UTC), "value": {"x": 2.0}},
                        {"time": datetime(2024, 6, 25, 12, 26, 15, 0, tzinfo=UTC), "value": {"x": 3.0}},
                    ],
                ],
            },
        )
        target_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {"time": datetime(2024, 6, 25, 12, 26, 3, 0, tzinfo=UTC), "value": {"x": 3}},
                        {"time": datetime(2024, 6, 25, 12, 26, 4, 0, tzinfo=UTC), "value": {"x": 4}},
                        {"time": datetime(2024, 6, 25, 12, 26, 5, 0, tzinfo=UTC), "value": {"x": 5}},
                    ],
                    [
                        {"time": datetime(2024, 6, 25, 12, 26, 8, 0, tzinfo=UTC), "value": {"x": 8}},
                        {"time": datetime(2024, 6, 25, 12, 26, 9, 0, tzinfo=UTC), "value": {"x": 9}},
                        {"time": datetime(2024, 6, 25, 12, 26, 10, 0, tzinfo=UTC), "value": {"x": 10}},
                    ],
                    [
                        {"time": datetime(2024, 6, 25, 12, 26, 13, 0, tzinfo=UTC), "value": {"x": 13}},
                        {"time": datetime(2024, 6, 25, 12, 26, 14, 0, tzinfo=UTC), "value": {"x": 14}},
                    ],
                ],

                "counter": [
                        {"time": datetime(2024, 6, 25, 12, 26, 5, 0, tzinfo=UTC), "value": {"x": 1.0}},
                        {"time": datetime(2024, 6, 25, 12, 26, 10, 0, tzinfo=UTC), "value": {"x": 2.0}},
                        {"time": datetime(2024, 6, 25, 12, 26, 15, 0, tzinfo=UTC), "value": {"x": 3.0}},
                ],
            },
        )
        tf = SliceTimeSeries(counter_col="counter", dur_in_sec=duration)
        transformed_data = tf.apply(input_data)
        print(input_data.collect())
        print(target_data.collect())
        print(transformed_data.collect())
        assert_frame_equal(transformed_data, target_data)

# TODO: QUESTIONS
# Should a counter entry be removed if there is no other feature data available during that duration?



if __name__ == "__main__":
    unittest.main()
