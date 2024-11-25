import unittest

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.transforms import NoMatchingCategoryError, OneCold


class OneHotTransform(unittest.TestCase):
    def test_single(self) -> None:
        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        )
        transform = OneCold.from_dataframe(data_frame, ["a"])
        transformed_data = transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a_1": [0, 1, 1, 1],
                    "a_4": [1, 0, 1, 1],
                    "a_7": [1, 1, 0, 1],
                    "a_10": [1, 1, 1, 0],
                    "b": [2, 5, 8, 11],
                    "c": [3, 6, 9, 12],
                },
            ),
            check_column_order=False,
        )

    def test_multiple(self) -> None:
        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        )
        transform = OneCold.from_dataframe(data_frame, ["a", "b"])
        transformed_data = transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a_1": [0, 1, 1, 1],
                    "a_4": [1, 0, 1, 1],
                    "a_7": [1, 1, 0, 1],
                    "a_10": [1, 1, 1, 0],
                    "b_2": [0, 1, 1, 1],
                    "b_5": [1, 0, 1, 1],
                    "b_8": [1, 1, 0, 1],
                    "b_11": [1, 1, 1, 0],
                    "c": [3, 6, 9, 12],
                },
            ),
            check_column_order=False,
        )

    def test_given_categories(self) -> None:
        transform = OneCold({"a": [1, 7]})
        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        )
        transformed_data = transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a_1": [0, 1, 1, 1],
                    "a_7": [1, 1, 0, 1],
                    "b": [2, 5, 8, 11],
                    "c": [3, 6, 9, 12],
                },
            ),
            check_column_order=False,
        )

    def test_missing_category(self) -> None:
        transform = OneCold({"a": [1, 7]}, check_for_missing_categories=True)
        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        )

        with pytest.raises(NoMatchingCategoryError):
            transform.apply(
                data_frame,
            )


if __name__ == "__main__":
    unittest.main()
