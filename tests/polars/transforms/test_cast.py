import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import Cast


class CastTransform(unittest.TestCase):
    def test_all(self) -> None:
        transform = Cast(pl.Float64)

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2},
                {"a": 3, "b": 4},
                {"a": 5, "b": 6},
            ],
        ).lazy()
        transformed_data = transform(data_frame).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"a": 1.0, "b": 2.0},
                    {"a": 3.0, "b": 4.0},
                    {"a": 5.0, "b": 6.0},
                ],
            ),
        )

    def test_selected(self) -> None:
        transform = Cast(pl.Float64, features=["b"])

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2},
                {"a": 3, "b": 4},
                {"a": 5, "b": 6},
            ],
        ).lazy()
        transformed_data = transform(data_frame).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"a": 1, "b": 2.0},
                    {"a": 3, "b": 4.0},
                    {"a": 5, "b": 6.0},
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
