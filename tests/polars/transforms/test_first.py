import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import First


class FirstTransform(unittest.TestCase):
    def test_single_feature(self) -> None:
        dataset = pl.DataFrame(
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
        )

        transform = First("feature_a")
        transformed = transform.apply(dataset.lazy()).collect()

        assert_frame_equal(
            transformed,
            pl.concat(
                [
                    dataset,
                    pl.DataFrame(
                        {
                            "feature_a_first": [42, 44],
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
                        {"time": 1, "value": 43},
                    ],
                    [
                        {"time": 2, "value": 44},
                        {"time": 3, "value": 45},
                    ],
                ],
            },
        )

        transform = First("feature_a", replace=True)
        transformed = transform.apply(dataset.lazy()).collect()

        assert_frame_equal(
            transformed,
            pl.DataFrame(
                {
                    "feature_a": [42, 44],
                },
            ),
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
