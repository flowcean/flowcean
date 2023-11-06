import unittest
from numpy import ndarray
from agenc.learners.dummy_learner import DummyLearner


class TestDummyLearner(unittest.TestCase):
    def test_dummy_learner(self):
        dummy_learner = DummyLearner()

        # Training with some example data should not raise any exceptions
        dummy_learner.train(ndarray([1, 2, 3]), ndarray([4, 5, 6]))

        # Predictions from the DummyLearner should be an empty array
        predictions = dummy_learner.predict(ndarray([7, 8, 9]))
        self.assertEqual(predictions.shape, (0,))

if __name__ == "__main__":
    unittest.main()