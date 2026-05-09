from dataclasses import dataclass


@dataclass(frozen=True)
class SelectorFeatureConfig:
    """Configuration for selector feature construction.

    Args:
        state_features: State columns available to the selector.
        input_features: Raw input columns available to the selector.
        derivative_features: Derivative columns available to the selector.
        state_history: Number of previous state rows used for selector
            features.
        input_history: Number of previous input rows used for selector
            features.
        derivative_history: Number of previous derivative rows used for
            selector features.
        mode_history: Number of previous mode labels used for selector
            features.
    """

    state_features: tuple[str, ...] = ()
    input_features: tuple[str, ...] = ()
    derivative_features: tuple[str, ...] = ()
    state_history: int = 0
    input_history: int = 0
    derivative_history: int = 0
    mode_history: int = 0

    def required_columns(self) -> tuple[str, ...]:
        return (
            *self.state_features,
            *self.input_features,
            *self.derivative_features,
        )

    @property
    def max_history(self) -> int:
        return max(
            self.state_history,
            self.input_history,
            self.derivative_history,
            self.mode_history,
        )

    def validate(self) -> None:
        histories = (
            self.state_history,
            self.input_history,
            self.derivative_history,
            self.mode_history,
        )
        if any(value < 0 for value in histories):
            message = "selector history values must be non-negative"
            raise ValueError(message)

        if self.state_history and not self.state_features:
            message = (
                "state_history requires at least one configured state column"
            )
            raise ValueError(message)

        if self.input_history and not self.input_features:
            message = (
                "input_history requires at least one configured input column"
            )
            raise ValueError(message)

        if self.derivative_history and not self.derivative_features:
            message = (
                "derivative_history requires at least one configured "
                "derivative column"
            )
            raise ValueError(message)

        raw_columns = self.required_columns()
        if len(set(raw_columns)) != len(raw_columns):
            message = "duplicate selector columns are not allowed"
            raise ValueError(message)

        if not (
            self.state_features
            or self.input_features
            or self.derivative_features
            or self.mode_history
        ):
            message = "selector config must request at least one feature"
            raise ValueError(message)
