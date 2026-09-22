"""Local, stateless prediction with learned AALpy automata."""

from __future__ import annotations

from abc import abstractmethod
from typing import override

import polars as pl
from aalpy.automata import MealyMachine, MooreMachine

from flowcean.core import Model

from ._data import Symbol, read_traces


class _RPNIModel[Automaton: MealyMachine | MooreMachine](Model):
    def __init__(
        self,
        automaton: Automaton,
        *,
        input_name: str,
        input_type: pl.DataType,
        output_name: str,
        output_type: pl.DataType,
    ) -> None:
        self._automaton = automaton
        self._input_name = input_name
        self._input_type = input_type
        self._output_name = output_name
        self._output_type = output_type

    @property
    def automaton(self) -> Automaton:
        """The wrapped AALpy automaton (not copied; treat its graph as read-only).

        Prediction does not read or change the automaton's ``current_state``.
        """
        return self._automaton

    @override
    def _predict(
        self, input_features: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        inputs = read_traces(input_features)
        if inputs.name != self._input_name:
            msg = f"Expected input column {self._input_name!r}, got {inputs.name!r}."
            raise ValueError(msg)
        if inputs.symbol_type != self._input_type:
            msg = (
                f"Expected input symbols with dtype {self._input_type}, "
                f"got {inputs.symbol_type}."
            )
            raise ValueError(msg)
        return pl.DataFrame(
            pl.Series(
                self._output_name,
                [self._predict_trace(word) for word in inputs.traces],
                dtype=pl.List(self._output_type),
            ),
        ).lazy()

    @abstractmethod
    def _predict_trace(self, word: list[Symbol]) -> list[Symbol]: ...


class RPNIMealyModel(_RPNIModel[MealyMachine]):
    """A Flowcean model wrapping an AALpy Mealy machine by composition.

    Predictions require the training input column's name and scalar dtype. They
    have the training output column's name and scalar dtype, with one list of
    output symbols per input trace. Timestamps order the inputs but are not
    predicted. An empty input word produces an empty output word.
    Undefined transitions raise ``ValueError``; no completion is invented.
    """

    @override
    def _predict_trace(self, word: list[Symbol]) -> list[Symbol]:
        state = self.automaton.initial_state
        outputs = []
        for index, symbol in enumerate(word):
            if symbol not in state.transitions:
                msg = f"Undefined transition for input {symbol!r} at position {index}."
                raise ValueError(msg)
            outputs.append(state.output_fun[symbol])
            state = state.transitions[symbol]
        return outputs


class RPNIMooreModel(_RPNIModel[MooreMachine]):
    """A Flowcean model wrapping an AALpy Moore machine by composition.

    Predictions require the training input column's name and scalar dtype. They
    have the training output column's name and scalar dtype, with one list of
    output symbols per input trace. Every output word starts with
    the initial-state output, even for empty input words. Timestamps order the
    inputs but are not predicted. Undefined transitions raise ``ValueError``.
    """

    @override
    def _predict_trace(self, word: list[Symbol]) -> list[Symbol]:
        state = self.automaton.initial_state
        outputs = [state.output]
        for index, symbol in enumerate(word):
            if symbol not in state.transitions:
                msg = f"Undefined transition for input {symbol!r} at position {index}."
                raise ValueError(msg)
            state = state.transitions[symbol]
            outputs.append(state.output)
        return outputs
