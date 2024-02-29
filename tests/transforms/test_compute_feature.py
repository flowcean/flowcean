import unittest

import polars as pl
from agenc.transforms import ComputeFeature
from polars.testing import assert_frame_equal


class TestComputeFeatureTransform(unittest.TestCase):
    def test_compute_single(self) -> None:
        transform = ComputeFeature(
            expression="a*2",
            output_feature_name="b",
        )

        data_frame = pl.DataFrame(
            [
                {"a": 1},
                {"a": 4},
                {"a": 7},
                {"a": 10},
            ],
        )
        transformed_data = transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"a": 1, "b": 2},
                    {"a": 4, "b": 8},
                    {"a": 7, "b": 14},
                    {"a": 10, "b": 20},
                ],
            ),
        )

    def test_compute_multiple(self) -> None:
        transform = ComputeFeature(
            expression="a+b",
            output_feature_name="c",
        )

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2},
                {"a": 4, "b": 5},
                {"a": 7, "b": 8},
                {"a": 10, "b": 11},
            ],
        )
        transformed_data = transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"a": 1, "b": 2, "c": 3},
                    {"a": 4, "b": 5, "c": 9},
                    {"a": 7, "b": 8, "c": 15},
                    {"a": 10, "b": 11, "c": 21},
                ],
            ),
        )

    def test_compute_function(self) -> None:
        from math import pi, sin

        transform = ComputeFeature(
            expression="sin(a)",
            output_feature_name="b",
        )

        data_frame = pl.DataFrame(
            [
                {"a": 0},
                {"a": pi / 2},
                {"a": pi},
                {"a": pi * 3 / 2},
                {"a": 2 * pi},
            ],
        )
        transformed_data = transform(data_frame)

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                [
                    {"a": 0, "b": sin(0)},
                    {"a": pi / 2, "b": sin(pi / 2)},
                    {"a": pi, "b": sin(pi)},
                    {"a": pi * 3 / 2, "b": sin(pi * 3 / 2)},
                    {"a": 2 * pi, "b": sin(2 * pi)},
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
