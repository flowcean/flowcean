import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import And, Filter, Not, Or


class FilterTransform(unittest.TestCase):
    def test_simple_filter(self) -> None:
        transform = Filter("a > 2")

        data_frame = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a": [3, 4, 5],
                    "b": [3, 2, 1],
                },
            ),
            check_column_order=False,
        )

    def test_multiple_filter_and(self) -> None:
        transform = Filter(And(["(a > 2)", "b != 3"]))

        data_frame = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a": [4, 5],
                    "b": [2, 1],
                },
            ),
            check_column_order=False,
        )

    def test_multiple_filter_or(self) -> None:
        transform = Filter(Or(["(a < 2)", "b < 2"]))

        data_frame = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a": [1, 5],
                    "b": [5, 1],
                },
            ),
            check_column_order=False,
        )

    def test_nested_expr(self) -> None:
        transform = Filter(
            And(
                [
                    "a > 2",
                    Not(
                        Or(
                            [
                                "a == 4",
                                "b == 2",
                            ],
                        ),
                    ),
                ],
            ),
        )

        data_frame = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a": [3, 5],
                    "b": [3, 1],
                },
            ),
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
