from numpy import ndarray


from agenc.core import Learner


class DummyLearner(Learner):
    def train(self, inputs: ndarray, outputs: ndarray) -> None:
        # The dummy learner does nothing during training
        pass

    def predict(self, inputs: ndarray) -> ndarray:
        # The dummy learner returns an empty array as a placeholder for predictions
        return ndarray(0)
    

# Example usage:
if __name__ == "__main__":
    # Create an instance of the DummyLearner
    dummy_learner = DummyLearner()

    # Dummy training with no effect
    dummy_learner.train(ndarray([1, 2, 3]), ndarray([4, 5, 6]))

    # Make a prediction, which will return an empty array
    predictions = dummy_learner.predict(ndarray([7, 8, 9]))
    print("Predictions:", predictions)