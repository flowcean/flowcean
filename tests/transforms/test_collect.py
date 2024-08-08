import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.transforms import Collect


class CollectTransform(unittest.TestCase):
    def test_collect(self) -> None:
        transform = Collect()

        data_frame = pl.DataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
                {"a": 7, "b": 8, "c": 9},
                {"a": 10, "b": 11, "c": 12},
            ],
        )
        transformed_data = transform.transform(data_frame.lazy())

        assert_frame_equal(
            transformed_data,
            data_frame,
        )


if __name__ == "__main__":
    unittest.main()
