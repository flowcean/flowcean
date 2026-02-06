import random
from typing import Literal

from .discrete import Discrete
from .domain import Domain

Distribution = Literal["uniform", "normal"]


class Continuous(Domain):
    """A domain of continuous values.

    This domain describes a continuous distribution of values between a minimum
    and maximum value for a feature.
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
        """Initialize the uniform feature domain.

        Args:
            feature_name: The name of the feature the domain belongs to.
            min_value: The minimum value of the domain.
            max_value: The maximum value of the domain.
            distribution: The distribution of values inside the domain.
                Can be either "uniform" or "normal". Defaults to "uniform".
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
        """Get a random value from the domain.

        Returns:
            A random value uniformly distributed between min_value and
            max_value.
        """
        if self.distribution == "normal":
            # We ignore the min and max values for the normal distribution
            # They are only used to check the mean and when the domain is
            # converted to discrete.
            return self.rng.gauss(self.mean, self.stddev)
        return (
            self.min_value
            + (self.max_value - self.min_value) * self.rng.random()
        )

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def to_discrete(self, sampling_distance: float) -> Discrete:
        """Discretize the continuous domain into a discrete domain.

        Args:
            sampling_distance: The distance between two discrete values.

        Returns:
            A discrete domain with the same feature name and a list of
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
