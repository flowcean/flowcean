import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import TimeSeriesSlidingWindow


class TimeSeriesSlidingWindowTransform(unittest.TestCase):
    def test_single_ts(self) -> None:
        dataset = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {"time": 0, "value": 42},
                        {"time": 1, "value": 43},
                        {"time": 2, "value": 44},
                    ],
                    [
                        {"time": 0, "value": 17},
                        {"time": 1, "value": 16},
                        {"time": 2, "value": 15},
                    ],
                ],
            },
        )

        transform = TimeSeriesSlidingWindow(2)
        transformed = transform.apply(dataset.lazy()).collect()

        assert_frame_equal(
            transformed,
            pl.DataFrame(
                {
                    "feature_a": [
                        [
                            {"time": 0, "value": 42},
                            {"time": 1, "value": 43},
                        ],
                        [
                            {"time": 1, "value": 43},
                            {"time": 2, "value": 44},
                        ],
                        [
                            {"time": 0, "value": 17},
                            {"time": 1, "value": 16},
                        ],
                        [
                            {"time": 1, "value": 16},
                            {"time": 2, "value": 15},
                        ],
                    ],
                },
            ),
            check_column_order=False,
        )

    def test_multiple_features(self) -> None:
        dataset = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {"time": 0, "value": 42},
                        {"time": 1, "value": 43},
                        {"time": 2, "value": 44},
                    ],
                ],
                "feature_b": [
                    [
                        {"time": 0, "value": 17},
                        {"time": 1, "value": 16},
                        {"time": 2, "value": 15},
                    ],
                ],
                "feature_c": ["a"],
            },
        )

        transform = TimeSeriesSlidingWindow(
            2,
            features=["feature_a", "feature_b"],
        )
        transformed = transform.apply(dataset.lazy()).collect()

        assert_frame_equal(
            transformed,
            pl.DataFrame(
                {
                    "feature_a": [
                        [
                            {"time": 0, "value": 42},
                            {"time": 1, "value": 43},
                        ],
                        [
                            {"time": 1, "value": 43},
                            {"time": 2, "value": 44},
                        ],
                    ],
                    "feature_b": [
                        [
                            {"time": 0, "value": 17},
                            {"time": 1, "value": 16},
                        ],
                        [
                            {"time": 1, "value": 16},
                            {"time": 2, "value": 15},
                        ],
                    ],
                    "feature_c": ["a", "a"],
                },
            ),
            check_column_order=False,
        )

    def test_stride(self) -> None:
        dataset = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {"time": 0, "value": 42},
                        {"time": 1, "value": 43},
                        {"time": 2, "value": 44},
                        {"time": 3, "value": 45},
                    ],
                ],
            },
        )

        transform = TimeSeriesSlidingWindow(2, stride=2)
        transformed = transform.apply(dataset.lazy()).collect()

        assert_frame_equal(
            transformed,
            pl.DataFrame(
                {
                    "feature_a": [
                        [
                            {"time": 0, "value": 42},
                            {"time": 1, "value": 43},
                        ],
                        [
                            {"time": 2, "value": 44},
                            {"time": 3, "value": 45},
                        ],
                    ],
                },
            ),
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
