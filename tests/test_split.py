import random
import unittest

import numpy as np
import polars as pl
import pytest
from agenc.core.split import split, train_test_split


class TestSplit(unittest.TestCase):
    def setUp(self) -> None:
        data = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
        self.data_frame = pl.DataFrame(data)

    def test_split(self) -> None:
        splits = split(self.data_frame, [2, 2], shuffle=False)
        assert np.array_equal(splits[0].to_numpy(), np.array([[1, 5], [2, 6]]))
        assert np.array_equal(splits[1].to_numpy(), np.array([[3, 7], [4, 8]]))

    def test_random_split(self) -> None:
        random.seed(1)
        splits = split(self.data_frame, [2, 2], shuffle=True)
        assert np.array_equal(splits[0].to_numpy(), np.array([[3, 7], [4, 8]]))
        assert np.array_equal(splits[1].to_numpy(), np.array([[2, 6], [1, 5]]))

    def test_train_test_split(self) -> None:
        random.seed(0)
        train, test = train_test_split(self.data_frame, 0.5)
        assert np.array_equal(train.to_numpy(), np.array([[3, 7], [4, 8]]))
        assert np.array_equal(test.to_numpy(), np.array([[1, 5], [2, 6]]))

    def test_ratio_is_between_zero_one(self) -> None:
        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            train_test_split(self.data_frame, -1.0)
        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            train_test_split(self.data_frame, 1.1)


if __name__ == "__main__":
    unittest.main()
