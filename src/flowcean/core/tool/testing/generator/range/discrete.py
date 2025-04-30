import random
from collections.abc import Iterable, Iterator

from .feature_value import FeatureValue


class Discrete(FeatureValue, Iterable[tuple[str, float]]):
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

    def __len__(self) -> int:
        """Get the number of values in the range.

        Returns:
            The number of values in the range.
        """
        return len(self.values)

    def get_value(self) -> float:
        """Get a random value from the range.

        Returns:
            A random value uniformly distributed between min_value and
            max_value.
        """
        return self.rng.choice(self.values)

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """Iterate over the values of the range."""
        for value in self.values:
            yield (self.feature_name, value)
