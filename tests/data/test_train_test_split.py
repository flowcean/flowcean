import random
import unittest

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.environments.dataset import Dataset
from flowcean.environments.train_test_split import TrainTestSplit


class TestTrainTestSplit(unittest.TestCase):
    def test_split(self) -> None:
        dataset = Dataset(pl.DataFrame({"a": [1, 2, 3, 4, 5, 6]}))
        train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(dataset)

        expected_train = pl.DataFrame({"a": [1, 2, 3, 4]})
        assert_frame_equal(expected_train, train.get_data())

        expected_test = pl.DataFrame({"a": [5, 6]})
        assert_frame_equal(expected_test, test.get_data())

        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            TrainTestSplit(ratio=1.5, shuffle=False)

    def test_split_shuffle(self) -> None:
        random.seed(42)
        dataset = Dataset(pl.DataFrame({"a": [1, 2, 3, 4, 5, 6]}))
        train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(dataset)

        expected_train = pl.DataFrame({"a": [5, 4, 2, 1]})
        assert_frame_equal(expected_train, train.get_data())

        expected_test = pl.DataFrame({"a": [3, 6]})
        assert_frame_equal(expected_test, test.get_data())


if __name__ == "__main__":
    unittest.main()
