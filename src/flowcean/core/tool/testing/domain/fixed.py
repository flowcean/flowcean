from .discrete import Discrete


class Fixed(Discrete):
    """A domain with a single value.

    This domain contains a single fixed value for a feature.
    """

    def __init__(self, feature_name: str, value: float) -> None:
        """Initialize the fixed domain.

        Args:
            feature_name: The name of the feature the domain belongs to.
            value: The fixed value to return.
        """
        super().__init__(feature_name, [value])

    def get_value(self) -> float:
        """Get the fixed value."""
        return self.values[0]
