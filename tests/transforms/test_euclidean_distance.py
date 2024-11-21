import math
import unittest

import numpy as np
import polars as pl

from flowcean.transforms import EuclideanDistance


class ExplodeTransform(unittest.TestCase):
    def test_euclidean_distance_transform(self) -> None:
        input_data = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {"time": 1.0, "value": {"x": 1.0, "y": 1.0}},
                        {"time": 2.0, "value": {"x": 2.0, "y": 2.0}},
                        {"time": 3.0, "value": {"x": 3.0, "y": 3.0}},
                        {"time": 4.0, "value": {"x": 4.0, "y": 4.0}},
                    ]
                ],
                "feature_b": [
                    [
                        {"time": 1.0, "value": {"x": 1.0, "y": 2.0}},
                        {"time": 2.0, "value": {"x": 1.0, "y": 2.0}},
                        {"time": 3.0, "value": {"x": 2.0, "y": 3.0}},
                        {"time": 4.0, "value": {"x": 3.0, "y": 4.0}},
                    ]
                ],
            }
        )

        transform = EuclideanDistance(
            feature_a_name="feature_a",
            feature_b_name="feature_b",
            output_feature_name="euclidean_distance",
        )

        result = transform.transform(input_data)

        # Check columns
        self.assertEqual(
            result.columns,
            [
                "feature_a",
                "feature_b",
                "euclidean_distance",
            ],
        )

        # Compute expected distances manually
        expected_distances = [
            math.sqrt((1.0 - 1.0) ** 2 + (1.0 - 2.0) ** 2),  # First point
            math.sqrt((2.0 - 1.0) ** 2 + (2.0 - 2.0) ** 2),  # Second point
            math.sqrt((3.0 - 2.0) ** 2 + (3.0 - 3.0) ** 2),  # Third point
            math.sqrt((4.0 - 3.0) ** 2 + (4.0 - 4.0) ** 2),  # Fourth point
        ]

        # Extract computed distances
        try:
            # Try to extract distances directly as numeric values
            computed_distances = (
                result.select("euclidean_distance")
                .explode("euclidean_distance")
                .to_series()
                .to_list()
            )
        except Exception:
            try:
                # Try to extract distances by unnesting
                computed_distances = (
                    result.select("euclidean_distance")
                    .explode("euclidean_distance")
                    .unnest("euclidean_distance")
                    .select("value")
                    .to_series()
                    .to_list()
                )
            except Exception as e:
                # If both methods fail, print the result for debugging
                print("Result structure:", result)
                print("Result schema:", result.schema)
                raise RuntimeError(
                    "Could not extract distances from the result"
                ) from e

        # Perform lenient comparison
        self.assertEqual(
            len(computed_distances),
            len(expected_distances),
            "Number of distances doesn't match expected",
        )

        for computed, expected in zip(computed_distances, expected_distances):
            # Handle different possible return types
            if isinstance(computed, dict):
                computed = computed.get("value", computed)

            # Use math.isclose for more flexible comparison
            self.assertTrue(
                math.isclose(
                    float(computed), expected, rel_tol=1e-6, abs_tol=1e-6
                ),
                f"Distance mismatch: computed {computed} vs expected {expected}",
            )

    def test_euclidean_distance_empty_dataframe(self) -> None:
        # Create an empty DataFrame with the correct schema
        input_data = pl.DataFrame(
            {
                "feature_a": pl.Series(
                    dtype=pl.List(
                        pl.Struct(
                            {
                                "time": pl.Float64,
                                "value": pl.Struct(
                                    {"x": pl.Float64, "y": pl.Float64}
                                ),
                            }
                        )
                    )
                ),
                "feature_b": pl.Series(
                    dtype=pl.List(
                        pl.Struct(
                            {
                                "time": pl.Float64,
                                "value": pl.Struct(
                                    {"x": pl.Float64, "y": pl.Float64}
                                ),
                            }
                        )
                    )
                ),
            }
        )

        transform = EuclideanDistance(
            feature_a_name="feature_a",
            feature_b_name="feature_b",
            output_feature_name="euclidean_distance",
        )

        result = transform.transform(input_data)

        # Check columns
        self.assertEqual(
            result.columns,
            [
                "feature_a",
                "feature_b",
                "euclidean_distance",
            ],
        )

        # Check if result is empty
        self.assertTrue(result.is_empty())

    def test_euclidean_distance_single_point(self) -> None:
        input_data = pl.DataFrame(
            {
                "feature_a": [[{"time": 1.0, "value": {"x": 1.0, "y": 1.0}}]],
                "feature_b": [[{"time": 1.0, "value": {"x": 2.0, "y": 2.0}}]],
            }
        )

        transform = EuclideanDistance(
            feature_a_name="feature_a",
            feature_b_name="feature_b",
            output_feature_name="euclidean_distance",
        )

        result = transform.transform(input_data)

        # Check columns
        self.assertEqual(
            result.columns,
            [
                "feature_a",
                "feature_b",
                "euclidean_distance",
            ],
        )

        try:
            # Try to extract distance directly
            computed_distance = (
                result.select("euclidean_distance")
                .explode("euclidean_distance")
                .to_series()
                .to_list()[0]
            )
        except Exception:
            # Try to extract distance by unnesting
            computed_distance = (
                result.select("euclidean_distance")
                .explode("euclidean_distance")
                .unnest("euclidean_distance")
                .select("value")
                .to_series()
                .to_list()[0]
            )

        # Handle different possible return types
        if isinstance(computed_distance, dict):
            computed_distance = computed_distance.get(
                "value", computed_distance
            )

        # Flexible comparison
        self.assertTrue(
            math.isclose(
                float(computed_distance),
                math.sqrt(2),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ),
            f"Distance mismatch: computed {computed_distance} vs expected {math.sqrt(2)}",
        )


if __name__ == "__main__":
    unittest.main()
