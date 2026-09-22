"""Conversion between Flowcean trace columns and passive RPNI samples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import polars as pl

type Symbol = str | bool | int | float


@dataclass
class TraceColumn:
    name: str
    symbol_type: pl.DataType
    traces: list[list[Symbol]]


def read_traces(data: pl.DataFrame | pl.LazyFrame) -> TraceColumn:
    """Read exactly one list column, stably ordering timestamped samples."""
    frame = data.collect() if isinstance(data, pl.LazyFrame) else data
    if frame.width != 1:
        msg = "Expected exactly one sequence column."
        raise ValueError(msg)
    column = frame.to_series()
    dtype = column.dtype
    if not isinstance(dtype, pl.List):
        msg = f"Column {column.name!r} must contain lists of symbols or time/value structs."
        raise ValueError(msg)
    symbol_type = dtype.inner
    timestamped = isinstance(symbol_type, pl.Struct)
    if isinstance(symbol_type, pl.Struct):
        fields = {field.name: field.dtype for field in symbol_type.fields}
        if set(fields) != {"time", "value"} or not fields["time"].is_numeric():
            msg = "Timestamped sequences require exactly numeric 'time' and scalar 'value' fields."
            raise ValueError(msg)
        symbol_type = fields["value"]
    if not (
        symbol_type.is_integer()
        or symbol_type.is_float()
        or symbol_type in (pl.String, pl.Boolean)
    ):
        msg = "Symbols must be strings, booleans, integers, or finite floats."
        raise ValueError(msg)

    traces = []
    for index, trace in enumerate(column.to_list()):
        if trace is None:
            msg = f"Column {column.name!r}, row {index}: null traces are not supported."
            raise ValueError(msg)
        symbols = trace
        if timestamped:
            if any(
                sample is None
                or sample["time"] is None
                or not math.isfinite(sample["time"])
                for sample in trace
            ):
                msg = f"Column {column.name!r}, row {index}: timestamps must be finite and non-null."
                raise ValueError(msg)
            symbols = [
                sample["value"]
                for sample in sorted(trace, key=lambda sample: sample["time"])
            ]
        if any(
            symbol is None
            or (isinstance(symbol, float) and not math.isfinite(symbol))
            for symbol in symbols
        ):
            msg = f"Column {column.name!r}, row {index}: symbols must be finite and non-null."
            raise ValueError(msg)
        traces.append(symbols)
    return TraceColumn(column.name, cast("pl.DataType", symbol_type), traces)


def rpni_samples(
    inputs: TraceColumn,
    outputs: TraceColumn,
    automaton_type: Literal["mealy", "moore"],
) -> list[tuple[tuple[Symbol, ...], Symbol]]:
    """Expand full traces into prefix labels, including Moore's empty prefix."""
    if len(inputs.traces) != len(outputs.traces) or not inputs.traces:
        msg = "Training requires the same nonzero number of input and output traces."
        raise ValueError(msg)
    initial_output = int(automaton_type == "moore")
    samples: dict[tuple[Symbol, ...], Symbol] = {}
    for index, (word, labels) in enumerate(
        zip(inputs.traces, outputs.traces, strict=True),
    ):
        if len(labels) != len(word) + initial_output:
            requirement = (
                "len(output) == len(input) + 1 (including the initial-state output)"
                if initial_output
                else "len(output) == len(input)"
            )
            msg = f"{automaton_type.capitalize()} trace {index} requires {requirement}."
            raise ValueError(msg)
        for length, label in enumerate(labels, start=1 - initial_output):
            prefix = tuple(word[:length])
            if prefix in samples and samples[prefix] != label:
                msg = f"Conflicting outputs for input prefix {prefix!r}; RPNI requires deterministic traces."
                raise ValueError(msg)
            samples[prefix] = label
    if not samples:
        msg = "Mealy training requires at least one nonempty trace."
        raise ValueError(msg)
    return list(samples.items())
