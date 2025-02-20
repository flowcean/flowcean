import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars.transforms import Median, ToTimeSeries


class MedianTransform(unittest.TestCase):
    def test_single_feature(self) -> None:
        transform = Median("a")

        data_frame_a = pl.DataFrame(
            [
                {"t": 0, "a": 0},
                {"t": 1, "a": 5},
                {"t": 2, "a": 10},
            ],
        ).lazy()

        data_frame_b = pl.DataFrame(
            [
                {"t": 0, "a": -100},
                {"t": 1, "a": 50},
                {"t": 2, "a": 50},
            ],
        ).lazy()

        time_series_transform = ToTimeSeries(time_feature="t")
        time_series_data_frame_a = time_series_transform(data_frame_a)
        time_series_data_frame_b = time_series_transform(data_frame_b)

        transformed_data = transform(
            pl.concat(
                [
                    time_series_data_frame_a,
                    time_series_data_frame_b,
                ],
                how="vertical",
            ),
        ).collect()

        assert_frame_equal(
            transformed_data.select(pl.col("a_median")),
            pl.DataFrame(
                [
                    {"a_median": 5.0},
                    {"a_median": 50.0},
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
