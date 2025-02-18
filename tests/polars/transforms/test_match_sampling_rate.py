import unittest
from datetime import datetime, timezone

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import MatchSamplingRate


class TestMatchSamplingRate(unittest.TestCase):
    def test_1d_timeseries_data(self) -> None:
        transform = MatchSamplingRate(
            reference_feature_name="feature_a",
            feature_interpolation_map={
                "feature_b": "linear",
            },
        )

        data_frame = pl.DataFrame(
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
                            "value": {"x": 1.2},
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
                            "value": {"x": 2.4},
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
                            "value": {"x": 3.6},
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
                            "value": {"x": 4.8},
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
                                0,
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
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )

        transformed_data = transform(data_frame.lazy()).collect()

        expected_data = pl.DataFrame(
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
                            "value": {"x": 1.2},
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
                            "value": {"x": 2.4},
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
                            "value": {"x": 3.6},
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
                            "value": {"x": 4.8},
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
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"feature_b_x": 1.2},
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
                            "value": {"feature_b_x": 1.4},
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
                            "value": {"feature_b_x": 1.6},
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
                            "value": {"feature_b_x": 1.8},
                        },
                    ],
                ],
                "const": [1],
            },
        )
        assert_frame_equal(transformed_data, expected_data)

    def test_2d_timeseries_data(self) -> None:
        transform = MatchSamplingRate(
            reference_feature_name="feature_a",
            feature_interpolation_map={
                "feature_b": "linear",
            },
        )

        data_frame = pl.DataFrame(
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
                            "value": {"x": 1.2, "y": 1.0},
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
                            "value": {"x": 2.4, "y": 2.0},
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
                            "value": {"x": 3.6, "y": 3.0},
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
                            "value": {"x": 4.8, "y": 4.0},
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
                                0,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0, "y": 10.0},
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
                            "value": {"x": 2.0, "y": 20.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )

        transformed_data = transform(data_frame.lazy()).collect()

        expected_data = pl.DataFrame(
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
                            "value": {"x": 1.2, "y": 1.0},
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
                            "value": {"x": 2.4, "y": 2.0},
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
                            "value": {"x": 3.6, "y": 3.0},
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
                            "value": {"x": 4.8, "y": 4.0},
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
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"feature_b_x": 1.2, "feature_b_y": 12.0},
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
                            "value": {"feature_b_x": 1.4, "feature_b_y": 14.0},
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
                            "value": {"feature_b_x": 1.6, "feature_b_y": 16.0},
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
                            "value": {"feature_b_x": 1.8, "feature_b_y": 18.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )
        assert_frame_equal(transformed_data, expected_data)

    def test_multiple_features(self) -> None:
        transform = MatchSamplingRate(
            reference_feature_name="feature_a",
            feature_interpolation_map={
                "feature_b": "linear",
                "feature_c": "linear",
            },
        )

        data_frame = pl.DataFrame(
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
                            "value": {"x": 1.2, "y": 1.0},
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
                            "value": {"x": 2.4, "y": 2.0},
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
                            "value": {"x": 3.6, "y": 3.0},
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
                            "value": {"x": 4.8, "y": 4.0},
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
                                0,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0, "y": 10.0},
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
                            "value": {"x": 2.0, "y": 20.0},
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
                                0,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0, "y": 20.0},
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
                            "value": {"x": 4.0, "y": 40.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )

        transformed_data = transform(data_frame.lazy()).collect()

        expected_data = pl.DataFrame(
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
                            "value": {"x": 1.2, "y": 1.0},
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
                            "value": {"x": 2.4, "y": 2.0},
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
                            "value": {"x": 3.6, "y": 3.0},
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
                            "value": {"x": 4.8, "y": 4.0},
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
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"feature_b_x": 1.2, "feature_b_y": 12.0},
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
                            "value": {"feature_b_x": 1.4, "feature_b_y": 14.0},
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
                            "value": {"feature_b_x": 1.6, "feature_b_y": 16.0},
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
                            "value": {"feature_b_x": 1.8, "feature_b_y": 18.0},
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
                            "value": {"feature_c_x": 2.4, "feature_c_y": 24.0},
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
                            "value": {"feature_c_x": 2.8, "feature_c_y": 28.0},
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
                            "value": {"feature_c_x": 3.2, "feature_c_y": 32.0},
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
                            "value": {"feature_c_x": 3.6, "feature_c_y": 36.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )
        assert_frame_equal(transformed_data, expected_data)

    def test_multiple_rows(self) -> None:
        transform = MatchSamplingRate(
            reference_feature_name="feature_a",
            feature_interpolation_map={
                "feature_b": "linear",
            },
        )

        data_frame = pl.DataFrame(
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
                            "value": {"x": 1.2, "y": 1.0},
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
                            "value": {"x": 2.4, "y": 2.0},
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
                            "value": {"x": 3.6, "y": 3.0},
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
                            "value": {"x": 4.8, "y": 4.0},
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
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.2, "y": 1.0},
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
                            "value": {"x": 2.4, "y": 2.0},
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
                            "value": {"x": 3.6, "y": 3.0},
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
                            "value": {"x": 4.8, "y": 4.0},
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
                                0,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0, "y": 10.0},
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
                            "value": {"x": 2.0, "y": 20.0},
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
                                0,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0, "y": 10.0},
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
                            "value": {"x": 2.0, "y": 20.0},
                        },
                    ],
                ],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()
        expected_data = pl.DataFrame(
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
                            "value": {"x": 1.2, "y": 1.0},
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
                            "value": {"x": 2.4, "y": 2.0},
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
                            "value": {"x": 3.6, "y": 3.0},
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
                            "value": {"x": 4.8, "y": 4.0},
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
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.2, "y": 1.0},
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
                            "value": {"x": 2.4, "y": 2.0},
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
                            "value": {"x": 3.6, "y": 3.0},
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
                            "value": {"x": 4.8, "y": 4.0},
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
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"feature_b_x": 1.2, "feature_b_y": 12.0},
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
                            "value": {"feature_b_x": 1.4, "feature_b_y": 14.0},
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
                            "value": {"feature_b_x": 1.6, "feature_b_y": 16.0},
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
                            "value": {"feature_b_x": 1.8, "feature_b_y": 18.0},
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
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"feature_b_x": 1.2, "feature_b_y": 12.0},
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
                            "value": {"feature_b_x": 1.4, "feature_b_y": 14.0},
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
                            "value": {"feature_b_x": 1.6, "feature_b_y": 16.0},
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
                            "value": {"feature_b_x": 1.8, "feature_b_y": 18.0},
                        },
                    ],
                ],
            },
        )
        assert_frame_equal(transformed_data, expected_data)

    def test_float_time_format(self) -> None:
        transform = MatchSamplingRate(
            reference_feature_name="feature_a",
            feature_interpolation_map={
                "feature_b": "linear",
            },
        )

        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": 1.0,
                            "value": {"x": 1.2, "y": 1.0},
                        },
                        {
                            "time": 2.0,
                            "value": {"x": 2.4, "y": 2.0},
                        },
                        {
                            "time": 3.0,
                            "value": {"x": 3.6, "y": 3.0},
                        },
                        {
                            "time": 4.0,
                            "value": {"x": 4.8, "y": 4.0},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": 0.0,
                            "value": {"x": 1.0, "y": 10.0},
                        },
                        {
                            "time": 5.0,
                            "value": {"x": 2.0, "y": 20.0},
                        },
                    ],
                ],
            },
        )

        transformed_data = transform(data_frame.lazy()).collect()

        expected_data = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": 1.0,
                            "value": {"x": 1.2, "y": 1.0},
                        },
                        {
                            "time": 2.0,
                            "value": {"x": 2.4, "y": 2.0},
                        },
                        {
                            "time": 3.0,
                            "value": {"x": 3.6, "y": 3.0},
                        },
                        {
                            "time": 4.0,
                            "value": {"x": 4.8, "y": 4.0},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": 1.0,
                            "value": {"feature_b_x": 1.2, "feature_b_y": 12.0},
                        },
                        {
                            "time": 2.0,
                            "value": {"feature_b_x": 1.4, "feature_b_y": 14.0},
                        },
                        {
                            "time": 3.0,
                            "value": {"feature_b_x": 1.6, "feature_b_y": 16.0},
                        },
                        {
                            "time": 4.0,
                            "value": {"feature_b_x": 1.8, "feature_b_y": 18.0},
                        },
                    ],
                ],
            },
        )
        assert_frame_equal(transformed_data, expected_data)


if __name__ == "__main__":
    unittest.main()
