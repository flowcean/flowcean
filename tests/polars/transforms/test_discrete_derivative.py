import unittest
from typing import ClassVar

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import DiscreteDerivative


class DiscreteDerivativeTransform(unittest.TestCase):
    feature_a: ClassVar[list[list[dict[str, float]]]] = [
        [
            {
                "time": 0.0,
                "value": 1.0,
            },
            {
                "time": 1.0,
                "value": 2.0,
            },
            {
                "time": 1.5,
                "value": 3.0,
            },
        ],
        [
            {
                "time": 0.0,
                "value": 0.0,
            },
            {
                "time": 5.0,
                "value": 5.0,
            },
            {
                "time": 15.0,
                "value": 25.0,
            },
        ],
    ]

    def test_forward(self) -> None:
        transform = DiscreteDerivative("feature_a", method="forward")
        data_frame = pl.DataFrame(
            {
                "feature_a": self.feature_a,
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        expected_data = pl.DataFrame(
            {
                "feature_a": self.feature_a,
                "feature_a_derivative": [
                    [
                        {
                            "time": 0.0,
                            "value": 1.0,
                        },
                        {
                            "time": 1.0,
                            "value": 2.0,
                        },
                    ],
                    [
                        {
                            "time": 0.0,
                            "value": 1.0,
                        },
                        {
                            "time": 5.0,
                            "value": 2.0,
                        },
                    ],
                ],
            },
        )

        assert_frame_equal(
            transformed_data,
            expected_data,
            check_column_order=False,
        )

    def test_backward(self) -> None:
        transform = DiscreteDerivative("feature_a", method="backward")
        data_frame = pl.DataFrame(
            {
                "feature_a": self.feature_a,
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        expected_data = pl.DataFrame(
            {
                "feature_a": self.feature_a,
                "feature_a_derivative": [
                    [
                        {
                            "time": 1.0,
                            "value": 1.0,
                        },
                        {
                            "time": 1.5,
                            "value": 2.0,
                        },
                    ],
                    [
                        {
                            "time": 5.0,
                            "value": 1.0,
                        },
                        {
                            "time": 15.0,
                            "value": 2.0,
                        },
                    ],
                ],
            },
        )

        assert_frame_equal(
            transformed_data,
            expected_data,
            check_column_order=False,
        )

    def test_central(self) -> None:
        transform = DiscreteDerivative("feature_a", method="central")

        data_frame = pl.DataFrame(
            {
                "feature_a": self.feature_a,
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()

        expected_data = pl.DataFrame(
            {
                "feature_a": self.feature_a,
                "feature_a_derivative": [
                    [
                        {
                            "time": 1.0,
                            "value": 2 / 1.5,
                        },
                    ],
                    [
                        {
                            "time": 5.0,
                            "value": 25.0 / 15.0,
                        },
                    ],
                ],
            },
        )

        assert_frame_equal(
            transformed_data,
            expected_data,
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
