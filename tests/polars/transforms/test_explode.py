import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import Explode


class ExplodeTransform(unittest.TestCase):
    def test_explode(self) -> None:
        transform = Explode(features=["time", "feature_a", "feature_b"])

        data_frame = pl.DataFrame(
            {
                "time": [[0, 1, 2, 3], [0, 2, 4, 6]],
                "feature_a": [[2, 1, 7, 2], [3, 4, 1, 0]],
                "feature_b": [[9, 3, 5, 0], [8, 4, 7, 2]],
                "constant": [1, 2],
            },
        )
        transformed_data = transform(data_frame.lazy()).collect()
        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "time": [0, 1, 2, 3, 0, 2, 4, 6],
                    "feature_a": [2, 1, 7, 2, 3, 4, 1, 0],
                    "feature_b": [9, 3, 5, 0, 8, 4, 7, 2],
                    "constant": [1, 1, 1, 1, 2, 2, 2, 2],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
