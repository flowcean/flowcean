import unittest

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.polars import (
    FeatureLengthVaryError,
    Flatten,
    NoTimeSeriesFeatureError,
)


class FlattenTransform(unittest.TestCase):
    def test_flatten_all(self) -> None:
        flatten_transform = Flatten()

        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": 0,
                            "value": 1,
                        },
                        {
                            "time": 1,
                            "value": 2,
                        },
                    ],
                    [
                        {
                            "time": 0,
                            "value": 3,
                        },
                        {
                            "time": 1,
                            "value": 4,
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": 0,
                            "value": 5,
                        },
                        {
                            "time": 1,
                            "value": 6,
                        },
                    ],
                    [
                        {
                            "time": 0,
                            "value": 7,
                        },
                        {
                            "time": 1,
                            "value": 8,
                        },
                    ],
                ],
                "scalar": [1, 2],
            },
        )

        transformed_data = flatten_transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "feature_a_0": [1, 3],
                    "feature_a_1": [2, 4],
                    "feature_b_0": [5, 7],
                    "feature_b_1": [6, 8],
                    "scalar": [1, 2],
                },
            ),
            check_column_order=False,
        )

    def test_flatten_some(self) -> None:
        flatten_transform = Flatten(["feature_a"])

        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": 0,
                            "value": 1,
                        },
                        {
                            "time": 1,
                            "value": 2,
                        },
                    ],
                    [
                        {
                            "time": 0,
                            "value": 3,
                        },
                        {
                            "time": 1,
                            "value": 4,
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": 0,
                            "value": 5,
                        },
                        {
                            "time": 1,
                            "value": 6,
                        },
                    ],
                    [
                        {
                            "time": 0,
                            "value": 7,
                        },
                        {
                            "time": 1,
                            "value": 8,
                        },
                    ],
                ],
                "scalar": [1, 2],
            },
        )

        transformed_data = flatten_transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "feature_a_0": [1, 3],
                    "feature_a_1": [2, 4],
                    "feature_b": [
                        [
                            {
                                "time": 0,
                                "value": 5,
                            },
                            {
                                "time": 1,
                                "value": 6,
                            },
                        ],
                        [
                            {
                                "time": 0,
                                "value": 7,
                            },
                            {
                                "time": 1,
                                "value": 8,
                            },
                        ],
                    ],
                    "scalar": [1, 2],
                },
            ),
            check_column_order=False,
        )

    def test_flatten_no_timeseries_feature(self) -> None:
        flatten_transform = Flatten(["scalar"])

        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": 0,
                            "value": 1,
                        },
                        {
                            "time": 1,
                            "value": 2,
                        },
                    ],
                    [
                        {
                            "time": 0,
                            "value": 3,
                        },
                        {
                            "time": 1,
                            "value": 4,
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": 0,
                            "value": 5,
                        },
                        {
                            "time": 1,
                            "value": 6,
                        },
                    ],
                    [
                        {
                            "time": 0,
                            "value": 7,
                        },
                        {
                            "time": 1,
                            "value": 8,
                        },
                    ],
                ],
                "scalar": [1, 2],
            },
        )

        with pytest.raises(NoTimeSeriesFeatureError):
            flatten_transform(data_frame.lazy()).collect()

    def test_flatten_varying_length(self) -> None:
        flatten_transform = Flatten()

        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": 0,
                            "value": 1,
                        },
                        {
                            "time": 1,
                            "value": 2,
                        },
                    ],
                    [
                        {
                            "time": 0,
                            "value": 3,
                        },
                        {
                            "time": 1,
                            "value": 4,
                        },
                        {
                            "time": 2,
                            "value": 5,
                        },
                    ],
                ],
            },
        )

        with pytest.raises(FeatureLengthVaryError):
            flatten_transform(data_frame.lazy()).collect()


if __name__ == "__main__":
    unittest.main()
