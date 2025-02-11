import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import Rename


class RenameTransform(unittest.TestCase):
    def test_rename(self) -> None:
        transform = Rename({"a": "d", "b": "e"})

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
                    {"d": 1, "e": 2, "c": 3},
                    {"d": 4, "e": 5, "c": 6},
                    {"d": 7, "e": 8, "c": 9},
                    {"d": 10, "e": 11, "c": 12},
                ],
            ),
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
