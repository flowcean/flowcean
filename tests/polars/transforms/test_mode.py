import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import Mode


class ModeTransform(unittest.TestCase):
    def test_single_feature(self) -> None:
        dataset = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {"time": 0, "value": 42},
                        {"time": 1, "value": 43},
                        {"time": 2, "value": 43},
                    ],
                    [
                        {"time": 0, "value": 0.0},
                        {"time": 1, "value": 1.0},
                        {"time": 2, "value": 0.0},
                    ],
                ],
            },
        )

        transform = Mode("feature_a")
        transformed = transform.apply(dataset.lazy()).collect()

        assert_frame_equal(
            transformed,
            pl.concat(
                [
                    dataset,
                    pl.DataFrame(
                        {
                            "feature_a_mode": [43, 0],
                        },
                    ),
                ],
                how="horizontal",
            ),
            check_column_order=False,
        )

    def test_replace_single_feature(self) -> None:
        dataset = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {"time": 0, "value": 42},
                        {"time": 1, "value": 42},
                        {"time": 2, "value": 43},
                        {"time": 3, "value": 43},
                    ],
                    [
                        {"time": 0, "value": 0.0},
                        {"time": 1, "value": 1.0},
                        {"time": 2, "value": 0.0},
                    ],
                ],
            },
        )

        transform = Mode("feature_a", replace=True)
        transformed = transform.apply(dataset.lazy()).collect()

        assert_frame_equal(
            transformed,
            pl.DataFrame(
                {
                    "feature_a": [43, 0],
                },
            ),
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
