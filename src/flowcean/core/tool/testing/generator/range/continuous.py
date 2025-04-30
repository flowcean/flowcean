import random
from typing import Literal

from .discrete import Discrete
from .feature_value import ValueRange

Distribution = Literal["uniform", "normal"]


class Continuous(ValueRange):
    """A range of continuous values.

    This range describes a continuous distribution of values between a minimum
    and maximum value for the given feature.
    """

    rng: random.Random

    def __init__(
        self,
        feature_name: str,
        min_value: float,
        max_value: float,
        *,
        distribution: Distribution = "uniform",
        mean: float | None = None,
        stddev: float | None = None,
    ) -> None:
        """Initialize the uniform range.

        Args:
            feature_name: The name of the feature the range belongs to.
            min_value: The minimum value of the range.
            max_value: The maximum value of the range.
            distribution: The distribution of the range. Can be either
                "uniform" or "normal". Defaults to "uniform".
            mean: The mean of the normal distribution. Required if
                distribution is "normal".
            stddev: The standard deviation of the normal distribution.
                Required if distribution is "normal".
        """
        super().__init__(feature_name)
        if min_value >= max_value:
            msg = (
                f"min_value ({min_value}) must be less"
                "than max_value ({max_value})"
            )
            raise ValueError(
                msg,
            )
        self.distribution = distribution
        if self.distribution == "normal":
            if mean is None or stddev is None:
                msg = (
                    "mean and stddev must be provided for normal distribution"
                )
                raise ValueError(msg)
            if not (min_value <= mean <= max_value):
                msg = (
                    f"mean ({mean}) must be between min_value ({min_value})"
                    f"and max_value ({max_value})"
                )
                raise ValueError(msg)
            self.mean = mean
            self.stddev = stddev

        self.min_value = min_value
        self.max_value = max_value
        self.rng = random.Random()

    def get_value(self) -> float:
        """Get a random value from the range.

        Returns:
            A random value uniformly distributed between min_value and
            max_value.
        """
        if self.distribution == "normal":
            # We ignore the min and max values for the normal distribution
            # They are only used to check the mean and when the range is
            # converted to discrete.
            return self.rng.gauss(self.mean, self.stddev)
        return (
            self.min_value
            + (self.max_value - self.min_value) * self.rng.random()
        )

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def to_discrete(self, sampling_distance: float) -> Discrete:
        """Convert the range to a discrete range.

        Args:
            sampling_distance: The distance between two discrete values.

        Returns:
            A discrete range with the same feature name and a list of
            uniformly distributed values.
        """
        return Discrete(
            self.feature_name,
            [
                self.min_value + i * sampling_distance
                for i in range(
                    int((self.max_value - self.min_value) / sampling_distance)
                    + 1,
                )
            ],
        )
