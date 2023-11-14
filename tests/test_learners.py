import unittest

import numpy as np
from agenc.learners.dummy_learner import DummyLearner


class TestDummyLearner(unittest.TestCase):
    def test_dummy_learner(self) -> None:
        dummy_learner = DummyLearner()
        dummy_learner.train(np.array([1, 2, 3]), np.array([4, 5, 6]))
        predictions = dummy_learner.predict(np.array([7, 8, 9]))
        assert predictions.shape == (3,)


if __name__ == "__main__":
    unittest.main()
