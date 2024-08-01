import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.transforms import Resample


class ResampleTransform(unittest.TestCase):
    def test_simple(self) -> None:
        transform = Resample(1.0)
        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": 0,
                            "value": 1,
                        },
                        {
                            "time": 2,
                            "value": 2,
                        },
                    ],
                    [
                        {
                            "time": 0,
                            "value": 0,
                        },
                        {
                            "time": 3,
                            "value": 3,
                        },
                    ],
                ],
                "scalar": [1, 2],
            }
        )
        transformed_data = transform.transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "feature_a": [
                        [
                            {
                                "time": 0,
                                "value": 1,
                            },
                            {
                                "time": 1,
                                "value": 1.5,
                            },
                            {
                                "time": 2,
                                "value": 2,
                            },
                        ],
                        [
                            {
                                "time": 0,
                                "value": 0,
                            },
                            {
                                "time": 1,
                                "value": 1,
                            },
                            {
                                "time": 2,
                                "value": 2,
                            },
                            {
                                "time": 3,
                                "value": 3,
                            },
                        ],
                    ],
                    "scalar": [1, 2],
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
