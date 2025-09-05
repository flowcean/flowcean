import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import Pad


class PadTransform(unittest.TestCase):
    def test_pad(self) -> None:
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
                        {"time": 6, "value": 46},
                    ],
                ],
            },
        )

        transform = Pad(5, features="feature_a")
        transformed = transform.apply(dataset.lazy()).collect()

        print(transformed)
        assert_frame_equal(
            transformed,
            pl.DataFrame(
                {
                    "feature_a": [
                        [
                            {"time": 0, "value": 42},
                            {"time": 1, "value": 43},
                            {"time": 5, "value": 43},
                        ],
                        [
                            {"time": 2, "value": 44},
                            {"time": 3, "value": 45},
                            {"time": 6, "value": 46},
                        ],
                    ],
                },
            ),
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
