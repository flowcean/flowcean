import unittest
from datetime import datetime, timezone

import polars as pl
from custom_transforms.slice_time_series import SliceTimeSeries
from polars.testing import assert_frame_equal


class SliceTimeSeriesTransform(unittest.TestCase):
    def test_1d_timeseries_data(self) -> None:
        transform = SliceTimeSeries(
            counter_column="feature_b",
            deadzone=2,
        )
        input_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 4},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 5},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                6,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 6},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                7,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 7},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 8},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 9},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                11,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 11},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                12,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 12},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 13},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 14},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )
        expected_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 4},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 5},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 8},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 9},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 13},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 14},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                ],
                "const": [
                    1,
                    1,
                    1,
                ],
            },
        )
        transformed_data = transform.apply(input_data)
        assert_frame_equal(transformed_data, expected_data)

    def test_2d_timeseries_data(self) -> None:
        transform = SliceTimeSeries(
            counter_column="feature_b",
            deadzone=2,
        )

        input_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1, "y": -1},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2, "y": -2},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3, "y": -3},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 4, "y": -4},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 5, "y": -5},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                6,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 6, "y": -6},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                7,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 7, "y": -7},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 8, "y": -8},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 9, "y": -9},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10, "y": -10},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                11,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 11, "y": -11},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                12,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 12, "y": -12},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 13, "y": -13},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 14, "y": -14},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0, "y": -1.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0, "y": -2.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0, "y": -3.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )
        expected_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3, "y": -3},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 4, "y": -4},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 5, "y": -5},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 8, "y": -8},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 9, "y": -9},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10, "y": -10},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 13, "y": -13},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 14, "y": -14},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0, "y": -1.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0, "y": -2.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0, "y": -3.0},
                        },
                    ],
                ],
                "const": [
                    1,
                    1,
                    1,
                ],
            },
        )
        transformed_data = transform.apply(input_data)
        assert_frame_equal(transformed_data, expected_data)

    def test_multiple_timeseries_data(self) -> None:
        transform = SliceTimeSeries(
            counter_column="feature_b",
            deadzone=2,
        )
        input_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 4},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 5},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                6,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 6},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                7,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 7},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 8},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 9},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                11,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 11},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                12,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 12},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 13},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 14},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                ],
                "feature_c": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 20},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 30},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 40},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 50},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                6,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 60},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                7,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 70},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 80},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 90},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 100},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                11,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 110},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                12,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 120},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 130},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 140},
                        },
                    ],
                ],
                "const": [1],
            },
        )
        expected_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 4},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 5},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 8},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 9},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 13},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 14},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                ],
                "feature_c": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 30},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 40},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 50},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 80},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 90},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 100},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 130},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 140},
                        },
                    ],
                ],
                "const": [
                    1,
                    1,
                    1,
                ],
            },
        )
        transformed_data = transform.apply(input_data)
        assert_frame_equal(transformed_data, expected_data)


"""
    def test_multiple_const_data(self) -> None:
        transform = SliceTimeSeries(
            counter_column="feature_b",
            duration=2,
        )
        input_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 4},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 5},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                6,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 6},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                7,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 7},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 8},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 9},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                11,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 11},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                12,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 12},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 13},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 14},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )
        expected_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 4},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 5},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 8},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 9},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 13},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 14},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                ],
                "const": [
                    1,
                    1,
                    1,
                ],
            },
        )
        transformed_data = transform.apply(input_data)
        assert_frame_equal(transformed_data, expected_data)

    def test_multiple_feature_time_series(self) -> None:
        transform = SliceTimeSeries(
            counter_column="feature_b",
            duration=2,
        )
        input_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 4},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 5},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                6,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 6},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                7,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 7},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 8},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 9},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                11,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 11},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                12,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 12},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 13},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 14},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 20},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 30},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 40},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 50},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                6,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 60},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                7,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 70},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 80},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 90},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 100},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                11,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 110},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                12,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 120},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 130},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 140},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                ],
                "const": [
                    1,
                    2,
                ],
            },
        )
        schema = pl.Schema(
            [
                (
                    "feature_a",
                    pl.List(
                        pl.Struct(
                            {
                                "time": pl.Datetime(
                                    time_unit="us",
                                    time_zone="UTC",
                                ),
                                "value": pl.Struct({"x": pl.Int64}),
                            },
                        ),
                    ),
                ),
                (
                    "feature_b",
                    pl.List(
                        pl.Struct(
                            {
                                "time": pl.Datetime(
                                    time_unit="us",
                                    time_zone="UTC",
                                ),
                                "value": pl.Struct({"x": pl.Float64}),
                            },
                        ),
                    ),
                ),
                ("const", pl.Int64),
            ],
        )

        expected_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 4},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 5},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 8},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 9},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 10},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 13},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 14},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 20},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 30},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 40},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                7,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 70},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                8,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 80},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 90},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                12,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 120},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                13,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 130},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 140},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                9,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                27,
                                14,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                ],
                "const": [
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                ],
            },
            schema=schema,
        )
        transformed_data = transform.apply(input_data)
        assert_frame_equal(transformed_data, expected_data)

    def test_no_relevant_time_series_data(self) -> None:
        transform = SliceTimeSeries(
            counter_column="feature_b",
            duration=2,
        )
        input_data = pl.LazyFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                6,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 6},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                7,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 7},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                11,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 11},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                12,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 12},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                ],
                "const": [
                    1,
                ],
            },
        )
        # create scema for  list[struct[2]]
        schema = pl.Schema(
            [
                (
                    "feature_a",
                    pl.List(
                        pl.Struct(
                            {
                                "time": pl.Datetime(
                                    time_unit="us",
                                    time_zone="UTC",
                                ),
                                "value": pl.Struct({"x": pl.Int64}),
                            },
                        ),
                    ),
                ),
                (
                    "feature_b",
                    pl.List(
                        pl.Struct(
                            {
                                "time": pl.Datetime(
                                    time_unit="us",
                                    time_zone="UTC",
                                ),
                                "value": pl.Struct({"x": pl.Float64}),
                            },
                        ),
                    ),
                ),
                ("const", pl.Int64),
            ],
        )
        expected_data = pl.DataFrame(
            {
                "feature_a": [
                    [],
                    [],
                    [],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                10,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                    ],
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                15,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 3.0},
                        },
                    ],
                ],
                "const": [
                    1,
                    1,
                    1,
                ],
            },
            schema=schema,
        )
        transformed_data = transform.apply(input_data)
        assert_frame_equal(transformed_data.collect(), expected_data)
"""

if __name__ == "__main__":
    unittest.main()
