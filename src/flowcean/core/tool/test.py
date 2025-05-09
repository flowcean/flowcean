import polars as pl
import tqdm

from flowcean.core import Model
from flowcean.core.data import Data
from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.tool.testing.predicates import Predicate


def test_model(
    model: Model,
    test_data: IncrementalEnvironment,
    predicate: Predicate,
    *,
    show_progress: bool = False,
    stop_after: int = 1,
) -> None:
    """Test a model with the given test data and predicate.

    This function runs the model on the test data and checks if the
    predictions satisfy the given predicate. If any prediction does not
    satisfy the predicate, a TestFailed exception is raised.
    This exception contains the input data and prediction that failed the
    predicate and can be used as a counterexample.
    This method relies on the model's predict method to obtain a prediction.
    It does not utilize the model's type or internal structure to prove
    predicates.

    Args:
        model: The model to test.
        test_data: The test data to use for testing the model. This must only
            include input features passed to the model and *not* the targets.
        predicate: The predicate used to check the model's predictions.
        show_progress: Whether to show progress during testing.
            Defaults to False.
        stop_after: Number of tests that need to fail before stopping. Defaults
            to 1. If set to 0 or negative, all tests are run regardless of
            failures.

    Raises:
        TestFailed: If the model's prediction does not satisfy the
            predicate.
    """
    number_of_failures = 0
    failure_data: list[Data] = []
    failure_prediction: list[Data] = []
    # Run the model on the test data
    for input_data in (
        tqdm.tqdm(
            test_data,
            "Testing Model",
            total=test_data.num_steps(),
        )
        if show_progress
        else test_data
    ):
        prediction = model.predict(input_data)

        # Handle dataframes and lazyframes separately
        # Those may contain multiple rows / samples and need to be
        # sliced to get the individual samples for testing
        if isinstance(prediction, pl.LazyFrame | pl.DataFrame) and isinstance(
            input_data,
            pl.LazyFrame | pl.DataFrame,
        ):
            input_data_collected = input_data.lazy().collect()
            prediction = prediction.lazy().collect()

            test_inputs = [
                input_data_collected.slice(i, 1)
                for i in range(len(input_data_collected))
            ]
            predictions = [
                prediction.slice(i, 1) for i in range(len(prediction))
            ]

        else:
            test_inputs = [input_data]
            predictions = [prediction]

        # Check if the prediction satisfies the predicate
        for test_input, prediction in zip(
            test_inputs,
            predictions,
            strict=True,
        ):
            if not predicate(
                test_input,
                prediction,
            ):
                number_of_failures += 1
                failure_data.append(test_input)
                failure_prediction.append(prediction)

                if number_of_failures >= stop_after > 0:
                    break

    # If we got any failures at this point, raise an exception
    if number_of_failures > 0:
        raise TestFailed(failure_data, failure_prediction)


class TestFailed(Exception):
    """Test failed exception.

    This exception is raised when a test fails.
    This happens when a model's prediction does not satisfy the given
    predicate.
    """

    def __init__(self, input_data: list[Data], prediction: list[Data]) -> None:
        self.input_data = [data.lazy().collect() for data in input_data]
        self.prediction = [data.lazy().collect() for data in prediction]

        if len(self.input_data) != len(self.prediction):
            msg = (
                "Input data and prediction must have the same length. "
                f"Got {len(self.input_data)} and {len(self.prediction)}."
            )
            raise ValueError(msg)

        if len(self.input_data) == 1:
            message = (
                f"Test failed for input data: {self.input_data} "
                f"with prediction: {self.prediction}"
            )
        else:
            message = (
                "Test failed. The following input data and predictions"
                " did not fulfill the predicate:\n"
            )

            for data, pred in zip(
                self.input_data,
                self.prediction,
                strict=True,
            ):
                message += f"Input data: {data}\nPrediction: {pred}\n\n"

        super().__init__(message)
