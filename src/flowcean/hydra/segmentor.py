import polars as pl

from flowcean.core.learner import SupervisedIncrementalLearner
from flowcean.core.model import Model


class Segmentor:
    """A class that segments traces.

    Attributes:
        target: The list of target feature names.
        inputs: The list of input feature names.
        start_width: The starting width of the segmentation window.
        step_width: The step width for enlarging the segmentation window.
        threshold: The threshold for the segmentation.

    """

    def __init__(
        self,
        target: list[str],
        inputs: list[str],
        start_width: int = 10,
        step_width: int = 5,
        threshold: float = 0.01,
    ) -> None:
        self.start_width = start_width
        self.step_width = step_width
        self.target = target
        self.inputs = inputs
        self.threshold = threshold

    def segment(
        self,
        data_frame: pl.DataFrame,
        learner: SupervisedIncrementalLearner,
    ) -> Model | None:
        """Perform segmentation on the given data frame.

        Args:
            data_frame: The data frame to be segmented.
            learner: The learner to be used.

        Returns:
            The learned model for a new mode or None if
            segmentation fails.

        """
        window_size = self.start_width - self.step_width
        fit = None
        function = None
        if len(data_frame) < self.start_width:
            return learner.learn_incremental(
                pl.LazyFrame(data_frame[self.inputs]),
                pl.LazyFrame(data_frame[self.target]),
            )
        while fit is None or (
            fit < self.threshold and window_size < len(data_frame)
        ):
            window_size += self.step_width
            segment = data_frame.slice(0, window_size)
            function = learner.learn_incremental(
                pl.LazyFrame(segment[self.inputs]),
                pl.LazyFrame(segment[self.target]),
            )
            predictions = function.predict(segment[self.inputs]).collect()
            # mse = mean_squared_error(data[target], predictions)
            fit = (
                (segment[self.target[0]] - predictions[self.target[0]])
                .abs()
                .max()
            )

        return function
