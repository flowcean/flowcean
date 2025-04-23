from abc import ABC, abstractmethod

from flowcean.core.data import Data


class Predicate(ABC):
    """Base class for predicates.

    A predicate is a function that takes the prediction of an model and returns
    a boolean value indicating whether the prediction satisfies a certain
    condition. Predicates can be combined using logical operators (AND, OR,
    NOT) to create more complex predicates.
    """

    @abstractmethod
    def __call__(self, input_data: Data, prediction: Data) -> bool:
        """Evaluate the predicate on a model prediction.

        Args:
            input_data: The input data used to generate the prediction.
            prediction: The prediction to evaluate.

        Returns:
            bool: True if the prediction satisfies the predicate,
                False otherwise.
        """

    def __and__(self, other: "Predicate") -> "AndPredicate":
        """Combine two predicates with a logical AND operation.

        Args:
            other: The other predicate to combine with.

        Returns:
            Predicate: A new predicate that is the logical AND of this and
                other.
        """
        return AndPredicate(self, other)

    def __or__(self, other: "Predicate") -> "OrPredicate":
        """Combine two predicates with a logical OR operation.

        Args:
            other: The other predicate to combine with.

        Returns:
            Predicate: A new predicate that is the logical OR of this and
                other.
        """
        return OrPredicate(self, other)

    def __invert__(self) -> "NotPredicate":
        """Negate the predicate.

        Returns:
            Predicate: A new predicate that is the negation of the original.
        """
        return NotPredicate(self)


class AndPredicate(Predicate):
    """Combine multiple predicates with a logical AND operation."""

    def __init__(self, *predicates: Predicate) -> None:
        self.predicates = predicates

    def __call__(self, input_data: Data, prediction: Data) -> bool:
        return all(
            predicate(input_data, prediction) for predicate in self.predicates
        )


class OrPredicate(Predicate):
    """Combine multiple predicates with a logical OR operation."""

    def __init__(self, *predicates: Predicate) -> None:
        self.predicates = predicates

    def __call__(self, input_data: Data, prediction: Data) -> bool:
        return any(
            predicate(input_data, prediction) for predicate in self.predicates
        )


class NotPredicate(Predicate):
    """Negate a predicate."""

    def __init__(self, predicate: Predicate) -> None:
        self.predicate = predicate

    def __call__(self, input_data: Data, prediction: Data) -> bool:
        return not self.predicate(input_data, prediction)
