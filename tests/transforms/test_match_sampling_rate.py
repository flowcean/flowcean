import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.transforms import MatchSamplingRate


class MatchSamplingRateTransform(unittest.TestCase):
    def test_match_sampling_rate(self) -> None:
        transform = MatchSamplingRate(
            reference_timestamps="time_feature_a",
            feature_columns_with_timestamps={
                "feature_b": "time_feature_b",
            },
        )

        data_frame = pl.DataFrame(
            {
                "time_feature_a": [[0, 1, 2], [0, 1, 2]],
                "feature_a": [[2, 1, 7], [4, 1, 0]],
                "time_feature_b": [[0, 2], [0, 2]],
                "feature_b": [[10, 20], [20, 40]],
                "constant": [1, 2],
            },
        )
        transformed_data = transform(data_frame)
        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "time_feature_a": [[0, 1, 2], [0, 1, 2]],
                    "feature_a": [[2, 1, 7], [4, 1, 0]],
                    "time_feature_b": [[0, 1, 2], [0, 1, 2]],
                    "feature_b": [[10, 15, 20], [20, 30, 40]],
                    "constant": [1, 2],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
