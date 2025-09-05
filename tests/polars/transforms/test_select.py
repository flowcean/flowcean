import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import Select


class SelectTransform(unittest.TestCase):
    def test_simple(self) -> None:
        transform = Select(features=["a", "c"])

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        )
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"a": 1, "c": 3},
                    {"a": 4, "c": 6},
                    {"a": 7, "c": 9},
                    {"a": 10, "c": 12},
                ],
            ),
        )

    def test_regex(self) -> None:
        transform = Select(features=["a", "^c.*$"])

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3, "afoo": 42, "cbar": 46},
                {"a": 4, "b": 5, "c": 6, "afoo": 43, "cbar": 47},
                {"a": 7, "b": 8, "c": 9, "afoo": 44, "cbar": 48},
                {"a": 10, "b": 11, "c": 12, "afoo": 45, "cbar": 49},
            ],
        )
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"a": 1, "c": 3, "cbar": 46},
                    {"a": 4, "c": 6, "cbar": 47},
                    {"a": 7, "c": 9, "cbar": 48},
                    {"a": 10, "c": 12, "cbar": 49},
                ],
            ),
        )

    def test_numpy(self) -> None:
        transform = Select(
            (np.sqrt(pl.col("a") + pl.col("b")) / pl.col("c")).alias(
                "computed",
            ),
        )

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        )
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "computed": [
                        0.5773502691896257,
                        0.5,
                        0.43033148291193524,
                        0.38188130791298663,
                    ],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
