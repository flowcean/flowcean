import random

from .range import Range


class Uniform(Range):
    """A range that generates random values uniformly distributed between a minimum and maximum value."""

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
