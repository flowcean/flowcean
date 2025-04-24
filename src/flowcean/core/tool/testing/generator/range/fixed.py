from .range import Range


class Fixed(Range):
    """A fixed range with a single value.

    This range describes a fixed value for the given feature.
    """

    def __init__(self, feature_name: str, value: float) -> None:
        """Initialize the fixed range.

        Args:
            feature_name: The name of the feature the range belongs to.
            value: The fixed value to return.
        """
        super().__init__(feature_name)
        self.value = value

    def get_value(self) -> float:
        """Get the fixed value."""
        return self.value
