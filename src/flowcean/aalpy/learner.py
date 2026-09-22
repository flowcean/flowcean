"""Passive RPNI learners for deterministic Mealy and Moore machines."""

from __future__ import annotations

from typing import override

import polars as pl
from aalpy.automata import MealyMachine, MooreMachine
from aalpy.learning_algs import run_RPNI

from flowcean.core import SupervisedLearner

from ._data import read_traces, rpni_samples
from .model import RPNIMealyModel, RPNIMooreModel


class RPNIMealyLearner(SupervisedLearner):
    """Learn a Mealy machine from full input/output traces using AALpy RPNI.

    Inputs and outputs must each select exactly one list column, with one trace
    per row. Lists contain ordered scalar symbols (strings, booleans, integers,
    or finite floats). Nulls are not supported. Input and output words must have
    equal lengths.

    Prefix expansion is handled internally. Contradictory traces are rejected.
    AALpy's default input-incomplete learning is retained: prediction raises
    ``ValueError`` for undefined transitions rather than inventing outputs.
    """

    @override
    def learn(
        self,
        inputs: pl.DataFrame | pl.LazyFrame,
        outputs: pl.DataFrame | pl.LazyFrame,
    ) -> RPNIMealyModel:
        input_traces = read_traces(inputs)
        output_traces = read_traces(outputs)
        automaton = run_RPNI(
            rpni_samples(input_traces, output_traces, "mealy"),
            automaton_type="mealy",
            input_completeness=None,
            print_info=False,
        )
        if not isinstance(automaton, MealyMachine):
            msg = "AALpy did not return a Mealy machine."
            raise RuntimeError(msg)
        return RPNIMealyModel(
            automaton,
            input_name=input_traces.name,
            input_type=input_traces.symbol_type,
            output_name=output_traces.name,
            output_type=output_traces.symbol_type,
        )


class RPNIMooreLearner(SupervisedLearner):
    """Learn a Moore machine, including its initial-state output.

    Accepts the same ordered scalar-word representation as
    ``RPNIMealyLearner``, but every output word must contain exactly one more
    symbol than its input word. The first output labels the initial state;
    subsequent outputs label states reached after each input. In particular,
    an empty input word requires one output symbol.

    Prediction returns symbol lists including the initial output. Undefined
    transitions raise ``ValueError``; input completion is not enabled.
    """

    @override
    def learn(
        self,
        inputs: pl.DataFrame | pl.LazyFrame,
        outputs: pl.DataFrame | pl.LazyFrame,
    ) -> RPNIMooreModel:
        input_traces = read_traces(inputs)
        output_traces = read_traces(outputs)
        automaton = run_RPNI(
            rpni_samples(input_traces, output_traces, "moore"),
            automaton_type="moore",
            input_completeness=None,
            print_info=False,
        )
        if not isinstance(automaton, MooreMachine):
            msg = "AALpy did not return a Moore machine."
            raise RuntimeError(msg)
        return RPNIMooreModel(
            automaton,
            input_name=input_traces.name,
            input_type=input_traces.symbol_type,
            output_name=output_traces.name,
            output_type=output_traces.symbol_type,
        )
