import random

from .range import Range


class Discrete(Range):
    """A range of discrete values.

    This range describes a discrete distribution of values from the given set
    for the given feature.
    """

    rng: random.Random

    def __init__(
        self,
        feature_name: str,
        values: list[float],
    ) -> None:
        """Initialize the discrete range.

        Args:
            feature_name: The name of the feature the range belongs to.
            values: The list of values of the range.
        """
        super().__init__(feature_name)
        self.values = values

        self.rng = random.Random()

    def get_value(self) -> float:
        """Get a random value from the range.

        Returns:
            A random value uniformly distributed between min_value and
            max_value.
        """
        return self.rng.choice(self.values)

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)
