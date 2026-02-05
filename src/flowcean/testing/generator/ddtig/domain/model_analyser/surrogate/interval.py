from __future__ import annotations
from enum import Enum

class Interval():
    """
    Represents an interval for a specific feature within an equivalence class.

    Attributes
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
        min_value: int | float,
        max_value: int | float
    ) -> None:
        """
        Initializes an Interval object.

        Args:
            feature : Index of the feature to which the interval belongs.
            left_endpoint : Left endpoint type ('(' for open, '[' for closed).
            right_endpoint : Right endpoint type (')' for open, ']' for closed).
            min_value : Lower bound of the interval.
            max_value : Upper bound of the interval.
        """
        self.feature = feature
        self.left_endpoint = left_endpoint
        self.right_endpoint = right_endpoint
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self):
        """
        Returns a string representation of the interval.
        """
        return self.left_endpoint.value + \
            str(self.min_value) + ',' + \
            str(self.max_value) + \
            self.right_endpoint.value
    
    def is_closed(self) -> bool:
        """
        Checks if the interval is fully closed [a, b].

        Returns:
            True if both endpoints are closed.
        """        
        return (self.left_endpoint == IntervalEndpoint.LEFT_CLOSED and
                self.right_endpoint == IntervalEndpoint.RIGHT_CLOSED)
    
    def is_open(self) -> bool:
        """
        Checks if the interval is fully open (a, b).

        Returns:
            True if both endpoints are open.
        """        
        return (self.left_endpoint == IntervalEndpoint.LEFT_OPEN and
                self.right_endpoint == IntervalEndpoint.RIGHT_OPEN)
    
    def is_right_open(self) -> bool:
        """
        Checks if the interval is right-open [a, b).

        Returns:
            True if left is closed and right is open.
        """       
        return (self.left_endpoint == IntervalEndpoint.LEFT_CLOSED and
                self.right_endpoint == IntervalEndpoint.RIGHT_OPEN)
    
    def is_left_open(self) -> bool:
        """
        Checks if the interval is left-open (a, b].

        Returns:
            True if left is open and right is closed.
        """       
        return (self.left_endpoint == IntervalEndpoint.RIGHT_CLOSED and
                self.right_endpoint == IntervalEndpoint.LEFT_OPEN)

    @staticmethod
    def is_subset(intervalA: Interval,
                  intervalB: Interval) -> Interval | None:
        """
        Determines which interval is a subset of the other.

        Args:
            intervalA : First interval to compare.
            intervalB : Second interval to compare.

        Returns:
            The superset interval if one contains the other,
            otherwise None.
        """
        # Determine which interval is larger
        if (intervalA.min_value <= intervalB.min_value and 
            intervalA.max_value >= intervalB.max_value):
            interval_large = intervalA
            interval_small = intervalB
        elif (intervalA.min_value >= intervalB.min_value and 
            intervalA.max_value <= intervalB.max_value):
            interval_large = intervalB
            interval_small = intervalA
        else:
            return None
        
        # Case 1: Strict containment
        if (interval_large.min_value < interval_small.min_value and
            interval_large.max_value > interval_small.max_value):
            return interval_large
        
        # Case 2: Same lower bound, larger upper bound
        if (interval_large.min_value == interval_small.min_value and
            interval_large.max_value > interval_small.max_value):
            if (interval_small.left_endpoint == IntervalEndpoint.LEFT_OPEN):
                return interval_large
            elif (interval_large.left_endpoint == IntervalEndpoint.LEFT_CLOSED):
                return interval_large
            else:
                return None  

        # Case 3: Smaller lower bound, same upper bound 
        if (interval_large.min_value < interval_small.min_value and
            interval_large.max_value == interval_small.max_value):
            if (interval_small.right_endpoint == IntervalEndpoint.RIGHT_OPEN):
                return interval_large
            elif (interval_large.right_endpoint == IntervalEndpoint.RIGHT_CLOSED):
                return interval_large
            else:
                return None
        
        # Case 4: Same bounds
        if (intervalA.min_value == intervalB.min_value and
            intervalA.max_value == intervalB.max_value):
            if (intervalA.is_closed()):
                return intervalA
            if (intervalA.is_open()):
                return intervalB
            if (intervalB.is_closed()):
                return intervalB
            if (intervalB.is_open()):
                return intervalA
            if (intervalA.left_endpoint == intervalB.left_endpoint and
                intervalA.right_endpoint == intervalB.right_endpoint):
                return intervalA
        return None

class IntervalEndpoint(Enum):
    """
    Enum representing the types of interval endpoints.
    """
    LEFT_OPEN = '('
    LEFT_CLOSED = '['
    RIGHT_OPEN = ')'
    RIGHT_CLOSED = ']'
