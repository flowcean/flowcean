from dataclasses import dataclass


@dataclass(frozen=True)
class SelectorFeatureConfig:
    state_columns: tuple[str, ...] = ()
    input_columns: tuple[str, ...] = ()
    derivative_columns: tuple[str, ...] = ()
    state_history: int = 0
    input_history: int = 0
    derivative_history: int = 0
    mode_history: int = 0

    def required_columns(self) -> tuple[str, ...]:
        return (
            *self.state_columns,
            *self.input_columns,
            *self.derivative_columns,
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

        if self.state_history and not self.state_columns:
            message = (
                "state_history requires at least one configured state column"
            )
            raise ValueError(message)

        if self.input_history and not self.input_columns:
            message = (
                "input_history requires at least one configured input column"
            )
            raise ValueError(message)

        if self.derivative_history and not self.derivative_columns:
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
            self.state_columns
            or self.input_columns
            or self.derivative_columns
            or self.mode_history
        ):
            message = "selector config must request at least one feature"
            raise ValueError(message)
