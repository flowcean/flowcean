from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class HyDRATraceSchema:
    """Column schema for HyDRA trace-based learning.

    Args:
        time: Time column name.
        state: State column names used as model inputs.
        derivative: Derivative column names used as model outputs.
        inputs: Optional external input column names.

    HyDRA currently expects derivative columns to align with state columns and
    supports single-output training in ``HyDRALearner``.
    """

    time: str
    state: tuple[str, ...]
    derivative: tuple[str, ...]
    inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._validate_disjoint_columns()

    @property
    def input_features(self) -> tuple[str, ...]:
        return (self.time, *self.state, *self.inputs)

    def validate_input_features(self, input_features: Sequence[str]) -> None:
        if len(input_features) != len(self.input_features) or set(
            input_features,
        ) != set(self.input_features):
            message = "input_features must match trace schema input columns"
            raise ValueError(message)

    def validate_output_features(self, output_features: Sequence[str]) -> None:
        if list(output_features) != list(self.derivative):
            message = "output_features must match schema.derivative order"
            raise ValueError(message)

    def validate_state_derivative_width(self) -> None:
        if len(self.state) != len(self.derivative):
            message = "schema state and derivative widths must match"
            raise ValueError(message)

    def _validate_disjoint_columns(self) -> None:
        columns = (self.time, *self.state, *self.derivative, *self.inputs)
        if len(set(columns)) != len(columns):
            message = "schema columns must be disjoint"
            raise ValueError(message)
