from __future__ import annotations

from enum import Enum


class Interval:
    """Represents an interval for a specific feature.

    The interval belongs to one equivalence class.

    Attributes:
    ----------
    feature : int
        Index of the feature to which the interval belongs.

    left_endpoint : IntervalEndpoint
        Indicates whether the interval is left-open or left-closed.

    right_endpoint : IntervalEndpoint
        Indicates whether the interval is right-open or right-closed.

    min_value : int | float
        Lower bound of the interval.

    max_value : int | float
        Upper bound of the interval.
    """

    def __init__(
        self,
        feature: int,
        left_endpoint: IntervalEndpoint,
        right_endpoint: IntervalEndpoint,
        min_value: float,
        max_value: float,
    ) -> None:
        """Initializes an Interval object.

        Args:
            feature : Index of the feature to which the interval belongs.
            left_endpoint : Left endpoint type ('(' for open, '[' for closed).
            right_endpoint : Right endpoint type
                (')' for open, ']' for closed).
            min_value : Lower bound of the interval.
            max_value : Upper bound of the interval.
        """
        self.feature = feature
        self.left_endpoint = left_endpoint
        self.right_endpoint = right_endpoint
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self) -> str:
        """Returns a string representation of the interval."""
        return (
            self.left_endpoint.value
            + str(self.min_value)
            + ","
            + str(self.max_value)
            + self.right_endpoint.value
        )

    def is_closed(self) -> bool:
        """Checks if the interval is fully closed [a, b].

        Returns:
            True if both endpoints are closed.
        """
        return (
            self.left_endpoint == IntervalEndpoint.LEFT_CLOSED
            and self.right_endpoint == IntervalEndpoint.RIGHT_CLOSED
        )

    def is_open(self) -> bool:
        """Checks if the interval is fully open (a, b).

        Returns:
            True if both endpoints are open.
        """
        return (
            self.left_endpoint == IntervalEndpoint.LEFT_OPEN
            and self.right_endpoint == IntervalEndpoint.RIGHT_OPEN
        )

    def is_right_open(self) -> bool:
        """Checks if the interval is right-open [a, b).

        Returns:
            True if left is closed and right is open.
        """
        return (
            self.left_endpoint == IntervalEndpoint.LEFT_CLOSED
            and self.right_endpoint == IntervalEndpoint.RIGHT_OPEN
        )

    def is_left_open(self) -> bool:
        """Checks if the interval is left-open (a, b].

        Returns:
            True if left is open and right is closed.
        """
        return (
            self.left_endpoint == IntervalEndpoint.RIGHT_CLOSED
            and self.right_endpoint == IntervalEndpoint.LEFT_OPEN
        )

    @staticmethod
    def _order_by_bounds(
        interval_a: Interval,
        interval_b: Interval,
    ) -> tuple[Interval, Interval] | None:
        if (
            interval_a.min_value <= interval_b.min_value
            and interval_a.max_value >= interval_b.max_value
        ):
            return interval_a, interval_b
        if (
            interval_a.min_value >= interval_b.min_value
            and interval_a.max_value <= interval_b.max_value
        ):
            return interval_b, interval_a
        return None

    @staticmethod
    def _same_bounds_superset(
        interval_a: Interval,
        interval_b: Interval,
    ) -> Interval | None:
        if interval_a.is_closed():
            return interval_a
        if interval_a.is_open():
            return interval_b
        if interval_b.is_closed():
            return interval_b
        if interval_b.is_open():
            return interval_a
        if (
            interval_a.left_endpoint == interval_b.left_endpoint
            and interval_a.right_endpoint == interval_b.right_endpoint
        ):
            return interval_a
        return None

    @staticmethod
    def is_subset(
        interval_a: Interval, interval_b: Interval,
    ) -> Interval | None:
        """Determines which interval is a subset of the other.

        Args:
            interval_a : First interval to compare.
            interval_b : Second interval to compare.

        Returns:
            The superset interval if one contains the other,
            otherwise None.
        """
        ordered = Interval._order_by_bounds(interval_a, interval_b)
        if ordered is None:
            return None

        interval_large, interval_small = ordered

        # Case 1: Strict containment
        if (
            interval_large.min_value < interval_small.min_value
            and interval_large.max_value > interval_small.max_value
        ):
            return interval_large

        # Case 2: Same lower bound, larger upper bound
        if (
            interval_large.min_value == interval_small.min_value
            and interval_large.max_value > interval_small.max_value
        ):
            left_ok = (
                interval_small.left_endpoint == IntervalEndpoint.LEFT_OPEN
                or interval_large.left_endpoint
                == IntervalEndpoint.LEFT_CLOSED
            )
            return interval_large if left_ok else None

        # Case 3: Smaller lower bound, same upper bound
        if (
            interval_large.min_value < interval_small.min_value
            and interval_large.max_value == interval_small.max_value
        ):
            right_ok = (
                interval_small.right_endpoint == IntervalEndpoint.RIGHT_OPEN
                or interval_large.right_endpoint
                == IntervalEndpoint.RIGHT_CLOSED
            )
            return interval_large if right_ok else None

        # Case 4: Same bounds
        if (
            interval_a.min_value == interval_b.min_value
            and interval_a.max_value == interval_b.max_value
        ):
            return Interval._same_bounds_superset(interval_a, interval_b)
        return None


class IntervalEndpoint(Enum):
    """Enum representing the types of interval endpoints."""

    LEFT_OPEN = "("
    LEFT_CLOSED = "["
    RIGHT_OPEN = ")"
    RIGHT_CLOSED = "]"
