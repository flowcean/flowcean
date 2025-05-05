from abc import ABC, abstractmethod


class Domain(ABC):
    """An abstract base class for describing the value domain for a feature."""

    feature_name: str

    def __init__(
        self,
        feature_name: str,
    ) -> None:
        """Initialize the domain.

        Args:
            feature_name: The name of the feature the domain belongs to.
        """
        self.feature_name = feature_name

    @abstractmethod
    def get_value(self) -> float:
        """Get a random value from the domain for the feature."""

    def __call__(self) -> float:
        """Get a random value from the domain for the feature."""
        return self.get_value()

    def set_seed(self, seed: int) -> None:  # noqa: ARG002
        """Set the seed for the random number generator.

        Args:
            seed: The seed to set.
        """
        # This is a no-op by default, as the base class does not maintain
        # any state related to random number generation.
        # Subclasses that maintain state can override this method to set
        # the seed for their random number generator.
        return
