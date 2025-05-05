import random
from collections.abc import Iterable, Iterator

from .domain import Domain


class Discrete(Domain, Iterable[tuple[str, float]]):
    """A domain of discrete values.

    This domain describes a discrete set of values for a feature.
    """

    rng: random.Random

    def __init__(
        self,
        feature_name: str,
        values: list[float],
    ) -> None:
        """Initialize the discrete domain.

        Args:
            feature_name: The name of the feature the domain belongs to.
            values: The list of values of the domain.
        """
        super().__init__(feature_name)
        self.values = values

        self.rng = random.Random()

    def __len__(self) -> int:
        """Get the number of discrete values in the domain.

        Returns:
            The number of discrete values in the domain.
        """
        return len(self.values)

    def get_value(self) -> float:
        """Get a random value from the domain."""
        return self.rng.choice(self.values)

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """Iterate over the values of the range."""
        for value in self.values:
            yield (self.feature_name, value)
