import unittest
from datetime import UTC, datetime

import polars as pl

from flowcean.transforms import MatchSamplingRate


class TestMatchSamplingRate(unittest.TestCase):
    def test_match_sampling_rate(self) -> None:
        transform = MatchSamplingRate(
            reference_feature="feature_a",
            feature_columns={
                "feature_b": "linear",
            },
        )

        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 1, 0, tzinfo=UTC
                            ),
                            "value": {"x": 1.2},
                        },
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 2, 0, tzinfo=UTC
                            ),
                            "value": {"x": 2.4},
                        },
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 3, 0, tzinfo=UTC
                            ),
                            "value": {"x": 3.6},
                        },
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 4, 0, tzinfo=UTC
                            ),
                            "value": {"x": 4.8},
                        },
                    ]
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 0, 0, tzinfo=UTC
                            ),
                            "value": {"x": 1.0},
                        },
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 5, 0, tzinfo=UTC
                            ),
                            "value": {"x": 2.0},
                        },
                    ]
                ],
                "const": [1],
            }
        )
        print(f"original df: {data_frame}")

        transformed_data = transform.transform(data_frame)

        expected_data = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 1, 0, tzinfo=UTC
                            ),
                            "value": {"x": 1.2},
                        },
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 2, 0, tzinfo=UTC
                            ),
                            "value": {"x": 2.4},
                        },
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 3, 0, tzinfo=UTC
                            ),
                            "value": {"x": 3.6},
                        },
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 4, 0, tzinfo=UTC
                            ),
                            "value": {"x": 4.8},
                        },
                    ]
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 1, 0, tzinfo=UTC
                            ),
                            "value": {"x": 1.2},
                        },
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 2, 0, tzinfo=UTC
                            ),
                            "value": {"x": 1.4},
                        },
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 3, 0, tzinfo=UTC
                            ),
                            "value": {"x": 1.6},
                        },
                        {
                            "time": datetime(
                                2024, 6, 25, 12, 26, 4, 0, tzinfo=UTC
                            ),
                            "value": {"x": 1.8},
                        },
                    ]
                ],
                "const": [1],
            }
        )
        print(
            f"expected df: {expected_data.explode("feature_a", "feature_b")}"
        )
        print(
            f"transformed df: {transformed_data.explode("feature_a", "feature_b")}"
        )
        assert transformed_data.frame_equal(expected_data)


if __name__ == "__main__":
    unittest.main()
