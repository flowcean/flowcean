import random
import unittest

import polars as pl
from agenc.data.split import TrainTestSplit
from polars.testing import assert_frame_equal


class TestTrainTestSplit(unittest.TestCase):
    def test_split(self) -> None:
        dataset = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6]})
        split = TrainTestSplit(ratio=0.8, shuffle=False)
        train, test = split(dataset)

        assert len(train) == 4
        assert len(test) == 2

        expected_train = pl.DataFrame({"a": [1, 2, 3, 4]})
        assert_frame_equal(expected_train, train)

        expected_test = pl.DataFrame({"a": [5, 6]})
        assert_frame_equal(expected_test, test)

    def test_split_shuffle(self) -> None:
        random.seed(42)
        dataset = pl.DataFrame({"a": [1, 2, 3, 4, 5, 6]})
        split = TrainTestSplit(ratio=0.8, shuffle=True)
        train, test = split(dataset)

        assert len(train) == 4
        assert len(test) == 2

        expected_train = pl.DataFrame({"a": [5, 4, 2, 1]})
        assert_frame_equal(expected_train, train)

        expected_test = pl.DataFrame({"a": [3, 6]})
        assert_frame_equal(expected_test, test)


if __name__ == "__main__":
    unittest.main()
