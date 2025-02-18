import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import Lambda


class LambdaTransform(unittest.TestCase):
    def test_func(self) -> None:
        transform = Lambda(lambda df: df.clear())

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        ).lazy()
        transformed_data = transform(data_frame).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                data={
                    "a": [],
                    "b": [],
                    "c": [],
                },
                schema={
                    "a": pl.Int64,
                    "b": pl.Int64,
                    "c": pl.Int64,
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
