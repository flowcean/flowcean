from abc import ABC, abstractmethod


class Predicate[Input, Prediction](ABC):
    """A condition on model inputs and predictions, composable with logic."""

    @abstractmethod
    def __call__(self, input_data: Input, prediction: Prediction) -> bool:
        """Return whether the input and prediction satisfy the condition."""

    def __and__(
        self,
        other: "Predicate[Input, Prediction]",
    ) -> "AndPredicate[Input, Prediction]":
        return AndPredicate(self, other)

    def __or__(
        self,
        other: "Predicate[Input, Prediction]",
    ) -> "OrPredicate[Input, Prediction]":
        return OrPredicate(self, other)

    def __invert__(self) -> "NotPredicate[Input, Prediction]":
        return NotPredicate(self)


class AndPredicate[Input, Prediction](Predicate[Input, Prediction]):
    """Combine predicates with a logical AND operation."""

    def __init__(self, *predicates: Predicate[Input, Prediction]) -> None:
        self.predicates = predicates

    def __call__(self, input_data: Input, prediction: Prediction) -> bool:
        return all(
            predicate(input_data, prediction) for predicate in self.predicates
        )


class OrPredicate[Input, Prediction](Predicate[Input, Prediction]):
    """Combine predicates with a logical OR operation."""

    def __init__(self, *predicates: Predicate[Input, Prediction]) -> None:
        self.predicates = predicates

    def __call__(self, input_data: Input, prediction: Prediction) -> bool:
        return any(
            predicate(input_data, prediction) for predicate in self.predicates
        )


class NotPredicate[Input, Prediction](Predicate[Input, Prediction]):
    """Negate a predicate."""

    def __init__(self, predicate: Predicate[Input, Prediction]) -> None:
        self.predicate = predicate

    def __call__(self, input_data: Input, prediction: Prediction) -> bool:
        return not self.predicate(input_data, prediction)
