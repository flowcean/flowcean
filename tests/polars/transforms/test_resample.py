import unittest
from datetime import time

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import Resample


class ResampleTransform(unittest.TestCase):
    def test_simple(self) -> None:
        transform = Resample(1.0)
        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": time(second=0),
                            "value": 1,
                        },
                        {
                            "time": time(second=2),
                            "value": 2,
                        },
                    ],
                    [
                        {
                            "time": time(second=0),
                            "value": 0,
                        },
                        {
                            "time": time(second=3),
                            "value": 3,
                        },
                    ],
                ],
                "scalar": [1, 2],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        expected_data = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": time(second=0),
                            "value": 1.0,
                        },
                        {
                            "time": time(second=1),
                            "value": 1.5,
                        },
                        {
                            "time": time(second=2),
                            "value": 2.0,
                        },
                    ],
                    [
                        {
                            "time": time(second=0),
                            "value": 0.0,
                        },
                        {
                            "time": time(second=1),
                            "value": 1.0,
                        },
                        {
                            "time": time(second=2),
                            "value": 2.0,
                        },
                        {
                            "time": time(second=3),
                            "value": 3.0,
                        },
                    ],
                ],
                "scalar": [1, 2],
            },
        )

        assert_frame_equal(
            transformed_data,
            expected_data,
            check_column_order=False,
        )

    def test_different_sample_rates(self) -> None:
        transform = Resample({"feature_a": 1.0, "feature_b": 2.0})
        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": time(second=0),
                            "value": 1,
                        },
                        {
                            "time": time(second=2),
                            "value": 2,
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": time(second=0),
                            "value": 0,
                        },
                        {
                            "time": time(second=1),
                            "value": 1,
                        },
                        {
                            "time": time(second=2),
                            "value": 2,
                        },
                    ],
                ],
                "scalar": [42],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        expected_data = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": time(second=0),
                            "value": 1.0,
                        },
                        {
                            "time": time(second=1),
                            "value": 1.5,
                        },
                        {
                            "time": time(second=2),
                            "value": 2.0,
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": time(second=0),
                            "value": 0.0,
                        },
                        {
                            "time": time(second=2),
                            "value": 2.0,
                        },
                    ],
                ],
                "scalar": [42],
            },
        )

        assert_frame_equal(
            transformed_data,
            expected_data,
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
