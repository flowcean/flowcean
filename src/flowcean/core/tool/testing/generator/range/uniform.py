import random

from .discrete import Discrete
from .range import Range


class Uniform(Range):
    """A range of uniform distributed values.

    This range describes a uniform distribution of values between a minimum
    and maximum value for the given feature.
    """

    rng: random.Random

    def __init__(
        self,
        feature_name: str,
        min_value: float,
        max_value: float,
    ) -> None:
        """Initialize the uniform range.

        Args:
            feature_name: The name of the feature the range belongs to.
            min_value: The minimum value of the range.
            max_value: The maximum value of the range.
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
        self.min_value = min_value
        self.max_value = max_value
        self.rng = random.Random()

    def get_value(self) -> float:
        """Get a random value from the range.

        Returns:
            A random value uniformly distributed between min_value and
            max_value.
        """
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
