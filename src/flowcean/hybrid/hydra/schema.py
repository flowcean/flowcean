from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class HyDRATraceSchema:
    """Column roles for a learned derivative model and its rollout.

    Args:
        time: Time column name.
        state: State column names used as model inputs.
        derivative: Derivative column names used as model outputs, in the
            same order as their corresponding state columns.
        inputs: Optional external input column names.

    All column names must be distinct. ``input_features`` collects the time,
    state, and external input columns. Rollout requires equal numbers of
    state and derivative columns.
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
