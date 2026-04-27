# HyDRA Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add schema-driven HyDRA model simulation and state-trace comparison so a learned HyDRA model can be rolled out and compared with a ground-truth hybrid-system trace.

**Architecture:** Introduce `HyDRATraceSchema` as the single source of truth for trace column semantics, wire it through `HyDRALearner` into `HyDRAModel`, and add simulation methods on `HyDRAModel` that integrate learned derivatives with SciPy. Keep trace comparison in a small `flowcean.hydra.simulation` module and return the existing `flowcean.ode.Trace` type from model simulation.

**Tech Stack:** Python, Polars, NumPy, SciPy `solve_ivp`, pytest, Flowcean `Model`, HyDRA selector models, `flowcean.ode.Trace`.

**Git Note:** Do not create commits unless the user explicitly requests them. The checkpoint steps in this plan use `git status --short` instead of `git commit`.

---

## File Structure

- Create `src/flowcean/hydra/schema.py`: define `HyDRATraceSchema` and validation helpers for schema disjointness and feature coverage.
- Modify `src/flowcean/hydra/learner.py`: accept `trace_schema`, validate it against `learn(inputs, outputs)`, and pass it into `HyDRAModel`.
- Modify `src/flowcean/hydra/model.py`: store `trace_schema`, validate model feature coverage, add `predict_next_state(...)`, `simulate(...)`, and small private helpers for row construction, input coercion, sample-grid preparation, and interval mode selection.
- Create `src/flowcean/hydra/simulation.py`: define `StateTraceComparison` and `compare_state_traces(...)`.
- Modify `src/flowcean/hydra/__init__.py`: export `HyDRATraceSchema`, `StateTraceComparison`, and `compare_state_traces`.
- Create `tests/hydra/test_hydra_simulation.py`: focused tests for schema validation, learner wiring, simulation, comparison, and selector routing.

## Task 1: Trace Schema Type

**Files:**
- Create: `src/flowcean/hydra/schema.py`
- Create: `tests/hydra/test_hydra_simulation.py`

- [ ] **Step 1: Write failing schema validation tests**

Add this initial content to `tests/hydra/test_hydra_simulation.py`:

```python
import numpy as np
import polars as pl
import pytest

from flowcean.hydra.schema import HyDRATraceSchema


def test_hydra_trace_schema_rejects_duplicate_columns() -> None:
    with pytest.raises(ValueError, match="schema columns must be disjoint"):
        HyDRATraceSchema(
            time="t",
            state=("x",),
            derivative=("dx",),
            inputs=("x",),
        )


def test_hydra_trace_schema_validates_input_feature_coverage() -> None:
    schema = HyDRATraceSchema(
        time="t",
        state=("x",),
        derivative=("dx",),
        inputs=("u",),
    )

    schema.validate_input_features(["x", "t", "u"])

    with pytest.raises(ValueError, match="input_features must match trace schema"):
        schema.validate_input_features(["x", "t"])


def test_hydra_trace_schema_validates_output_feature_order() -> None:
    schema = HyDRATraceSchema(
        time="t",
        state=("x", "v"),
        derivative=("dx", "dv"),
    )

    schema.validate_output_features(["dx", "dv"])

    with pytest.raises(ValueError, match="output_features must match schema.derivative"):
        schema.validate_output_features(["dv", "dx"])
```

- [ ] **Step 2: Run schema tests to verify they fail**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_hydra_trace_schema_rejects_duplicate_columns tests/hydra/test_hydra_simulation.py::test_hydra_trace_schema_validates_input_feature_coverage tests/hydra/test_hydra_simulation.py::test_hydra_trace_schema_validates_output_feature_order -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'flowcean.hydra.schema'`.

- [ ] **Step 3: Implement `HyDRATraceSchema`**

Create `src/flowcean/hydra/schema.py`:

```python
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class HyDRATraceSchema:
    time: str
    state: tuple[str, ...]
    derivative: tuple[str, ...]
    inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._validate_disjoint_columns()

    @property
    def input_columns(self) -> tuple[str, ...]:
        return (self.time, *self.state, *self.inputs)

    def validate_input_features(self, input_features: Sequence[str]) -> None:
        if set(input_features) != set(self.input_columns):
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
```

- [ ] **Step 4: Run schema tests to verify they pass**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_hydra_trace_schema_rejects_duplicate_columns tests/hydra/test_hydra_simulation.py::test_hydra_trace_schema_validates_input_feature_coverage tests/hydra/test_hydra_simulation.py::test_hydra_trace_schema_validates_output_feature_order -v
```

Expected: PASS for all three tests.

- [ ] **Step 5: Checkpoint status**

Run:

```bash
git status --short
```

Expected: shows new `src/flowcean/hydra/schema.py` and `tests/hydra/test_hydra_simulation.py`; do not commit unless the user explicitly requests it.

## Task 2: Learner Schema Wiring

**Files:**
- Modify: `src/flowcean/hydra/learner.py:46-204`
- Modify: `src/flowcean/hydra/model.py:21-35`
- Test: `tests/hydra/test_hydra_simulation.py`

- [ ] **Step 1: Add failing learner wiring tests**

Append this code to `tests/hydra/test_hydra_simulation.py`:

```python
from flowcean.hydra.learner import HyDRALearner
from tests.hydra.test_hydra import AlwaysZeroLearner


def test_hydra_learner_passes_trace_schema_to_model() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=1.0,
        start_width=1,
        step_width=1,
        trace_schema=schema,
    )

    model = learner.learn(
        inputs=pl.DataFrame({"t": [0.0, 1.0], "x": [0.0, 1.0]}).lazy(),
        outputs=pl.DataFrame({"dx": [0.0, 0.0]}).lazy(),
    )

    assert model.trace_schema == schema


def test_hydra_learner_validates_schema_input_coverage() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=1.0,
        trace_schema=schema,
    )

    with pytest.raises(ValueError, match="input_features must match trace schema"):
        learner.learn(
            inputs=pl.DataFrame({"x": [0.0, 1.0]}).lazy(),
            outputs=pl.DataFrame({"dx": [0.0, 0.0]}).lazy(),
        )


def test_hydra_learner_rejects_multi_dimensional_schema_for_now() -> None:
    schema = HyDRATraceSchema(
        time="t",
        state=("x", "v"),
        derivative=("dx", "dv"),
    )
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=1.0,
        trace_schema=schema,
    )

    with pytest.raises(ValueError, match="single-output"):
        learner.learn(
            inputs=pl.DataFrame(
                {"t": [0.0, 1.0], "x": [0.0, 1.0], "v": [1.0, 1.0]},
            ).lazy(),
            outputs=pl.DataFrame(
                {"dx": [0.0, 0.0], "dv": [1.0, 1.0]},
            ).lazy(),
        )
```

- [ ] **Step 2: Run learner wiring tests to verify they fail**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_hydra_learner_passes_trace_schema_to_model tests/hydra/test_hydra_simulation.py::test_hydra_learner_validates_schema_input_coverage tests/hydra/test_hydra_simulation.py::test_hydra_learner_rejects_multi_dimensional_schema_for_now -v
```

Expected: FAIL with `TypeError` for unexpected `trace_schema` or missing `trace_schema` attribute on the model.

- [ ] **Step 3: Wire schema through `HyDRALearner`**

Modify imports and constructor in `src/flowcean/hydra/learner.py`:

```python
from flowcean.hydra.schema import HyDRATraceSchema
```

Add the class attribute:

```python
    trace_schema: HyDRATraceSchema | None
```

Change `HyDRALearner.__init__` signature and assignment:

```python
        selector_learner: HybridDecisionTreeLearner | None = None,
        callback: HyDRACallback | None = None,
        trace_schema: HyDRATraceSchema | None = None,
    ) -> None:
```

```python
        self.selector_learner = selector_learner
        self.callback = callback
        self.trace_schema = trace_schema
```

At the start of `learn(...)`, collect columns before `_initialize_traces(...)` and validate them:

```python
        input_columns = inputs.collect_schema().names()
        output_columns = outputs.collect_schema().names()
        self._validate_trace_schema_for_learning(input_columns, output_columns)
```

The start of `learn(...)` should have this order:

```python
        input_columns = inputs.collect_schema().names()
        output_columns = outputs.collect_schema().names()
        self._validate_trace_schema_for_learning(input_columns, output_columns)
        traces = self._initialize_traces(inputs, outputs)
```

There should be no second assignment to `input_columns` or `output_columns` below `_initialize_traces(...)`.

Add this method inside `HyDRALearner` before `_discover_modes(...)`:

```python
    def _validate_trace_schema_for_learning(
        self,
        input_columns: list[str],
        output_columns: list[str],
    ) -> None:
        if self.trace_schema is None:
            return
        self.trace_schema.validate_input_features(input_columns)
        self.trace_schema.validate_output_features(output_columns)
        self.trace_schema.validate_state_derivative_width()
        if len(self.trace_schema.derivative) != 1:
            message = (
                "HyDRALearner currently supports single-output training only."
            )
            raise ValueError(message)
```

Pass the schema into the model construction:

```python
        model = HyDRAModel(
            learned_modes.models,
            input_features=input_columns,
            output_features=output_columns,
            selector=selector,
            trace_schema=self.trace_schema,
        )
```

- [ ] **Step 4: Add schema storage on `HyDRAModel`**

Modify imports in `src/flowcean/hydra/model.py`:

```python
from flowcean.hydra.schema import HyDRATraceSchema
```

Change `HyDRAModel.__init__` signature and body:

```python
    def __init__(
        self,
        modes: list[Model],
        *,
        input_features: list[str],
        output_features: list[str],
        selector: HybridDecisionTreeModel | None = None,
        trace_schema: HyDRATraceSchema | None = None,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.input_features = input_features
        self.output_features = output_features
        self.selector = selector
        self.trace_schema = trace_schema
        self._validate_trace_schema()
```

Add this private method inside `HyDRAModel` before `_predict(...)`:

```python
    def _validate_trace_schema(self) -> None:
        if self.trace_schema is None:
            return
        self.trace_schema.validate_input_features(self.input_features)
        self.trace_schema.validate_output_features(self.output_features)
```

- [ ] **Step 5: Run learner wiring tests to verify they pass**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_hydra_learner_passes_trace_schema_to_model tests/hydra/test_hydra_simulation.py::test_hydra_learner_validates_schema_input_coverage tests/hydra/test_hydra_simulation.py::test_hydra_learner_rejects_multi_dimensional_schema_for_now -v
```

Expected: PASS for all three tests.

- [ ] **Step 6: Checkpoint status**

Run:

```bash
git status --short
```

Expected: shows schema, learner, model, and test changes; do not commit unless the user explicitly requests it.

## Task 3: Single-Mode Next-State Integration

**Files:**
- Modify: `src/flowcean/hydra/model.py`
- Test: `tests/hydra/test_hydra_simulation.py`

- [ ] **Step 1: Add failing single-mode integration tests**

Append this code to `tests/hydra/test_hydra_simulation.py`:

```python
from flowcean.core import Model
from flowcean.hydra.model import HyDRAModel


class LinearDerivativeModel(Model):
    def __init__(self, output_columns: tuple[str, ...]) -> None:
        self.output_columns = output_columns

    def _predict(
        self,
        input_features: pl.LazyFrame | pl.DataFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        values: dict[str, list[float]] = {}
        if self.output_columns == ("dx",):
            values["dx"] = [2.0] * frame.height
        else:
            values["dx"] = [1.0] * frame.height
            values["dv"] = [-0.5] * frame.height
        return pl.DataFrame(values).lazy()


def test_hydra_model_predict_next_state_integrates_single_mode_derivative() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
        trace_schema=schema,
    )

    next_state = model.predict_next_state([1.0], t=0.0, dt=0.5)

    assert np.allclose(next_state, [2.0])


def test_hydra_model_predict_next_state_requires_schema() -> None:
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
    )

    with pytest.raises(ValueError, match="trace_schema"):
        model.predict_next_state([1.0], t=0.0, dt=0.5)


def test_hydra_model_predict_next_state_validates_state_width() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
        trace_schema=schema,
    )

    with pytest.raises(ValueError, match="state dimension"):
        model.predict_next_state([1.0, 2.0], t=0.0, dt=0.5)
```

- [ ] **Step 2: Run integration tests to verify they fail**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_hydra_model_predict_next_state_integrates_single_mode_derivative tests/hydra/test_hydra_simulation.py::test_hydra_model_predict_next_state_requires_schema tests/hydra/test_hydra_simulation.py::test_hydra_model_predict_next_state_validates_state_width -v
```

Expected: FAIL with `AttributeError: 'HyDRAModel' object has no attribute 'predict_next_state'`.

- [ ] **Step 3: Implement `predict_next_state(...)` helpers**

Modify imports in `src/flowcean/hydra/model.py`:

```python
from collections.abc import Iterable

import numpy as np
from scipy.integrate import solve_ivp

from flowcean.ode import InputStream
```

Add these methods inside `HyDRAModel` after `predict_with_diagnostics(...)`:

```python
    def predict_next_state(
        self,
        state: Iterable[float],
        *,
        t: float,
        dt: float,
        input_stream: InputStream | None = None,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_step: float | None = None,
    ) -> np.ndarray:
        schema = self._require_trace_schema()
        state_array = self._coerce_state(state)
        if len(self.output_features) != state_array.shape[0]:
            message = "Derivative output width must match the state dimension."
            raise ValueError(message)
        if dt <= 0 or not np.isfinite(dt):
            message = "dt must be a finite positive value."
            raise ValueError(message)
        mode_id = self._select_mode_id(t, state_array, input_stream)
        flow_model = self.modes[mode_id]

        solve_kwargs = {
            "fun": lambda time, values: self._predict_derivative(
                flow_model,
                float(time),
                values,
                input_stream,
            ),
            "t_span": (float(t), float(t + dt)),
            "y0": state_array,
            "t_eval": [float(t + dt)],
            "rtol": rtol,
            "atol": atol,
        }
        if max_step is not None:
            solve_kwargs["max_step"] = max_step

        result = solve_ivp(**solve_kwargs)
        if not result.success:
            message = f"HyDRA simulation failed: {result.message}"
            raise RuntimeError(message)
        return np.asarray(result.y[:, -1], dtype=float)

    def _require_trace_schema(self) -> HyDRATraceSchema:
        if self.trace_schema is None:
            message = "HyDRAModel simulation requires trace_schema."
            raise ValueError(message)
        return self.trace_schema

    def _coerce_state(self, state: Iterable[float]) -> np.ndarray:
        schema = self._require_trace_schema()
        values = np.asarray(list(state), dtype=float)
        if values.ndim != 1:
            message = "state must be a 1D vector."
            raise ValueError(message)
        if values.shape[0] != len(schema.state):
            message = "state dimension must match schema.state."
            raise ValueError(message)
        return values

    def _select_mode_id(
        self,
        t: float,
        state: np.ndarray,
        input_stream: InputStream | None,
    ) -> int:
        if len(self.modes) == 1:
            return 0
        if self.selector is None:
            message = "HyDRAModel simulation requires a selector for multiple modes."
            raise NotImplementedError(message)
        if self.selector.feature_config.max_history > 0:
            message = "HyDRA simulation does not support selector history features yet."
            raise NotImplementedError(message)
        frame = self._build_simulation_frame(t, state, input_stream)
        prediction = self.predict_with_diagnostics(frame)
        if not prediction.selector_results:
            message = "HyDRA selector produced no mode for simulation."
            raise ValueError(message)
        mode_id = prediction.selector_results[0].mode_id
        if mode_id is None:
            message = "HyDRA selector did not return a ready mode."
            raise ValueError(message)
        if mode_id < 0 or mode_id >= len(self.modes):
            message = f"selector predicted unknown mode ID {mode_id}"
            raise ValueError(message)
        return mode_id

    def _predict_derivative(
        self,
        flow_model: Model,
        t: float,
        state: np.ndarray,
        input_stream: InputStream | None,
    ) -> np.ndarray:
        frame = self._build_simulation_frame(t, state, input_stream)
        prediction = flow_model.predict(frame.select(self.input_features)).collect()
        derivative = prediction.select(self.output_features).to_numpy()[0]
        derivative = np.asarray(derivative, dtype=float)
        if derivative.shape[0] != state.shape[0]:
            message = "Derivative output width must match the state dimension."
            raise ValueError(message)
        return derivative

    def _build_simulation_frame(
        self,
        t: float,
        state: np.ndarray,
        input_stream: InputStream | None,
    ) -> pl.DataFrame:
        schema = self._require_trace_schema()
        row: dict[str, float] = {schema.time: float(t)}
        row.update(
            {column: float(value) for column, value in zip(schema.state, state, strict=True)},
        )
        inputs = self._simulation_inputs(t, input_stream)
        row.update(
            {column: float(value) for column, value in zip(schema.inputs, inputs, strict=True)},
        )
        return pl.DataFrame([row]).select(self.input_features)

    def _simulation_inputs(
        self,
        t: float,
        input_stream: InputStream | None,
    ) -> np.ndarray:
        schema = self._require_trace_schema()
        if not schema.inputs:
            return np.empty((0,), dtype=float)
        if input_stream is None:
            message = "input_stream is required for schema.inputs."
            raise ValueError(message)
        values = np.asarray(input_stream(float(t)), dtype=float)
        if values.ndim != 1:
            message = "input_stream must return a 1D array."
            raise ValueError(message)
        if values.shape[0] != len(schema.inputs):
            message = "input_stream width must match schema.inputs."
            raise ValueError(message)
        return values
```

- [ ] **Step 4: Run integration tests to verify they pass**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_hydra_model_predict_next_state_integrates_single_mode_derivative tests/hydra/test_hydra_simulation.py::test_hydra_model_predict_next_state_requires_schema tests/hydra/test_hydra_simulation.py::test_hydra_model_predict_next_state_validates_state_width -v
```

Expected: PASS for all three tests.

- [ ] **Step 5: Checkpoint status**

Run:

```bash
git status --short
```

Expected: shows model and test changes; do not commit unless the user explicitly requests it.

## Task 4: HyDRA Trace Rollout

**Files:**
- Modify: `src/flowcean/hydra/model.py`
- Test: `tests/hydra/test_hydra_simulation.py`

- [ ] **Step 1: Add failing rollout tests**

Append this code to `tests/hydra/test_hydra_simulation.py`:

```python
def test_hydra_model_simulate_returns_trace_on_sample_grid() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
        trace_schema=schema,
    )

    trace = model.simulate((0.0, 1.0), [0.0], sample_dt=0.5)

    assert np.allclose(trace.t, [0.0, 0.5, 1.0])
    assert np.allclose(trace.x[:, 0], [0.0, 1.0, 2.0])
    assert trace.location.tolist() == ["mode_0", "mode_0", "mode_0"]
    assert trace.events == ()
    assert trace.dx is None


def test_hydra_model_simulate_captures_inputs_when_requested() -> None:
    schema = HyDRATraceSchema(
        time="t",
        state=("x",),
        derivative=("dx",),
        inputs=("u",),
    )
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x", "u"],
        output_features=["dx"],
        trace_schema=schema,
    )

    trace = model.simulate(
        (0.0, 1.0),
        [0.0],
        sample_times=[0.0, 0.5, 1.0],
        input_stream=lambda t: np.array([t + 1.0], dtype=float),
    )

    assert trace.u is not None
    assert np.allclose(trace.u[:, 0], [1.0, 1.5, 2.0])


def test_hydra_model_simulate_rejects_missing_sample_grid() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
        trace_schema=schema,
    )

    with pytest.raises(ValueError, match="sample_times or sample_dt"):
        model.simulate((0.0, 1.0), [0.0])
```

- [ ] **Step 2: Run rollout tests to verify they fail**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_hydra_model_simulate_returns_trace_on_sample_grid tests/hydra/test_hydra_simulation.py::test_hydra_model_simulate_captures_inputs_when_requested tests/hydra/test_hydra_simulation.py::test_hydra_model_simulate_rejects_missing_sample_grid -v
```

Expected: FAIL with `AttributeError: 'HyDRAModel' object has no attribute 'simulate'`.

- [ ] **Step 3: Implement `simulate(...)` and sample-grid helpers**

Modify imports in `src/flowcean/hydra/model.py`:

```python
from flowcean.ode import Trace
```

Add these methods inside `HyDRAModel` after `predict_next_state(...)`:

```python
    def simulate(
        self,
        t_span: tuple[float, float],
        x0: Iterable[float],
        *,
        input_stream: InputStream | None = None,
        capture_inputs: bool | None = None,
        sample_times: Iterable[float] | None = None,
        sample_dt: float | None = None,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_step: float | None = None,
    ) -> Trace:
        times = self._prepare_simulation_times(t_span, sample_times, sample_dt)
        state = self._coerce_state(x0)
        states = [state.copy()]
        locations = ["mode_0" if len(self.modes) == 1 else f"mode_{self._select_mode_id(float(times[0]), state, input_stream)}"]

        for start, end in zip(times[:-1], times[1:], strict=True):
            mode_id = self._select_mode_id(float(start), state, input_stream)
            state = self.predict_next_state(
                state,
                t=float(start),
                dt=float(end - start),
                input_stream=input_stream,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            states.append(state.copy())
            locations.append(f"mode_{mode_id}")

        captured_inputs = None
        if self._should_capture_inputs(capture_inputs, input_stream):
            captured_inputs = np.vstack(
                [self._simulation_inputs(float(time), input_stream) for time in times],
            )

        return Trace(
            t=times,
            x=np.vstack(states),
            location=np.asarray(locations, dtype=object),
            events=(),
            u=captured_inputs,
            dx=None,
        )

    def _prepare_simulation_times(
        self,
        t_span: tuple[float, float],
        sample_times: Iterable[float] | None,
        sample_dt: float | None,
    ) -> np.ndarray:
        if sample_times is None and sample_dt is None:
            message = "Provide either sample_times or sample_dt."
            raise ValueError(message)
        if sample_times is not None and sample_dt is not None:
            message = "Provide either sample_times or sample_dt, not both."
            raise ValueError(message)
        if sample_times is not None:
            times = np.asarray(list(sample_times), dtype=float)
            if times.size == 0:
                message = "sample_times must contain at least one time."
                raise ValueError(message)
            if not np.all(np.isfinite(times)):
                message = "sample_times must be finite."
                raise ValueError(message)
        else:
            if sample_dt is None or not np.isfinite(sample_dt):
                message = "sample_dt must be finite."
                raise ValueError(message)
            if sample_dt <= 0:
                message = "sample_dt must be positive."
                raise ValueError(message)
            times = np.arange(t_span[0], t_span[1], sample_dt, dtype=float)
            endpoint_atol = float(np.finfo(float).eps * max(1.0, abs(float(t_span[1]))))
            if times.size and np.isclose(times[-1], t_span[1], rtol=0.0, atol=endpoint_atol):
                times[-1] = float(t_span[1])
            else:
                times = np.append(times, float(t_span[1]))
        if np.any(np.diff(times) <= 0):
            message = "sample_times must be strictly increasing."
            raise ValueError(message)
        if float(times[0]) != float(t_span[0]) or float(times[-1]) != float(t_span[1]):
            message = "sample_times must start and end at t_span boundaries."
            raise ValueError(message)
        return times

    def _should_capture_inputs(
        self,
        capture_inputs: bool | None,
        input_stream: InputStream | None,
    ) -> bool:
        if capture_inputs is None:
            return input_stream is not None
        if capture_inputs and input_stream is None:
            message = "capture_inputs=True requires an input_stream."
            raise ValueError(message)
        return capture_inputs
```

- [ ] **Step 4: Run rollout tests to verify they pass**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_hydra_model_simulate_returns_trace_on_sample_grid tests/hydra/test_hydra_simulation.py::test_hydra_model_simulate_captures_inputs_when_requested tests/hydra/test_hydra_simulation.py::test_hydra_model_simulate_rejects_missing_sample_grid -v
```

Expected: PASS for all three tests.

- [ ] **Step 5: Checkpoint status**

Run:

```bash
git status --short
```

Expected: shows model and test changes; do not commit unless the user explicitly requests it.

## Task 5: Trace Comparison Helper and Public Exports

**Files:**
- Create: `src/flowcean/hydra/simulation.py`
- Modify: `src/flowcean/hydra/__init__.py`
- Test: `tests/hydra/test_hydra_simulation.py`

- [ ] **Step 1: Add failing comparison and export tests**

Append this code to `tests/hydra/test_hydra_simulation.py`:

```python
from flowcean.ode import Trace


def test_compare_state_traces_reports_error_metrics() -> None:
    from flowcean.hydra import compare_state_traces

    reference = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[0.0], [2.0]], dtype=float),
        location=np.array(["a", "a"], dtype=object),
        events=(),
    )
    predicted = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[0.5], [1.0]], dtype=float),
        location=np.array(["mode_0", "mode_0"], dtype=object),
        events=(),
    )

    comparison = compare_state_traces(reference, predicted)

    assert np.allclose(comparison.absolute_error[:, 0], [0.5, 1.0])
    assert comparison.mae == pytest.approx(0.75)
    assert comparison.rmse == pytest.approx(np.sqrt((0.25 + 1.0) / 2.0))
    assert comparison.max_error == pytest.approx(1.0)


def test_compare_state_traces_allows_tiny_time_grid_drift() -> None:
    from flowcean.hydra import compare_state_traces

    reference = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[0.0], [1.0]], dtype=float),
        location=np.array(["a", "a"], dtype=object),
        events=(),
    )
    predicted = Trace(
        t=np.array([0.0, 1.0 + 5e-13], dtype=float),
        x=np.array([[0.0], [1.0]], dtype=float),
        location=np.array(["mode_0", "mode_0"], dtype=object),
        events=(),
    )

    comparison = compare_state_traces(reference, predicted)

    assert comparison.max_error == pytest.approx(0.0)


def test_compare_state_traces_rejects_mismatched_time_grid() -> None:
    from flowcean.hydra import compare_state_traces

    reference = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[0.0], [1.0]], dtype=float),
        location=np.array(["a", "a"], dtype=object),
        events=(),
    )
    predicted = Trace(
        t=np.array([0.0, 1.1], dtype=float),
        x=np.array([[0.0], [1.0]], dtype=float),
        location=np.array(["mode_0", "mode_0"], dtype=object),
        events=(),
    )

    with pytest.raises(ValueError, match="time grids"):
        compare_state_traces(reference, predicted)


def test_hydra_package_exports_simulation_api() -> None:
    from flowcean import hydra

    assert "HyDRATraceSchema" in hydra.__all__
    assert "StateTraceComparison" in hydra.__all__
    assert "compare_state_traces" in hydra.__all__
```

- [ ] **Step 2: Run comparison tests to verify they fail**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_compare_state_traces_reports_error_metrics tests/hydra/test_hydra_simulation.py::test_compare_state_traces_allows_tiny_time_grid_drift tests/hydra/test_hydra_simulation.py::test_compare_state_traces_rejects_mismatched_time_grid tests/hydra/test_hydra_simulation.py::test_hydra_package_exports_simulation_api -v
```

Expected: FAIL with import errors for `compare_state_traces` and missing package exports.

- [ ] **Step 3: Implement comparison helper**

Create `src/flowcean/hydra/simulation.py`:

```python
from dataclasses import dataclass

import numpy as np

from flowcean.ode import Trace


@dataclass(frozen=True)
class StateTraceComparison:
    absolute_error: np.ndarray
    mae: float
    rmse: float
    max_error: float


def compare_state_traces(reference: Trace, predicted: Trace) -> StateTraceComparison:
    if reference.t.shape != predicted.t.shape or not np.allclose(
        reference.t,
        predicted.t,
        rtol=0.0,
        atol=1e-12,
    ):
        message = "Trace time grids must match."
        raise ValueError(message)
    if reference.x.shape != predicted.x.shape:
        message = "Trace state shapes must match."
        raise ValueError(message)

    absolute_error = np.abs(reference.x - predicted.x)
    return StateTraceComparison(
        absolute_error=absolute_error,
        mae=float(np.mean(absolute_error)),
        rmse=float(np.sqrt(np.mean(np.square(reference.x - predicted.x)))),
        max_error=float(np.max(absolute_error)) if absolute_error.size else 0.0,
    )
```

- [ ] **Step 4: Export public simulation API**

Modify `src/flowcean/hydra/__init__.py`:

```python
__all__ = (
    "HyDRALearner",
    "HyDRALivePlotCallback",
    "HyDRAModel",
    "HyDRAReplay",
    "HyDRAReplayEmitter",
    "HyDRAStep",
    "HyDRATraceSchema",
    "HybridDecisionTreeLearner",
    "HybridDecisionTreeModel",
    "ModePredictionResult",
    "SelectorFeatureConfig",
    "StateTraceComparison",
    "compare_state_traces",
    "plot_hydra_replay_step",
)

from .learner import HyDRALearner
from .live_plot import HyDRALivePlotCallback
from .model import HyDRAModel
from .plotting import plot_hydra_replay_step
from .replay import HyDRAReplay, HyDRAReplayEmitter, HyDRAStep
from .schema import HyDRATraceSchema
from .selector import (
    HybridDecisionTreeLearner,
    HybridDecisionTreeModel,
    ModePredictionResult,
    SelectorFeatureConfig,
)
from .simulation import StateTraceComparison, compare_state_traces
```

- [ ] **Step 5: Run comparison tests to verify they pass**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_compare_state_traces_reports_error_metrics tests/hydra/test_hydra_simulation.py::test_compare_state_traces_allows_tiny_time_grid_drift tests/hydra/test_hydra_simulation.py::test_compare_state_traces_rejects_mismatched_time_grid tests/hydra/test_hydra_simulation.py::test_hydra_package_exports_simulation_api -v
```

Expected: PASS for all four tests.

- [ ] **Step 6: Checkpoint status**

Run:

```bash
git status --short
```

Expected: shows simulation helper and export changes; do not commit unless the user explicitly requests it.

## Task 6: Multi-Dimensional Manual Simulation and Selector Routing

**Files:**
- Modify: `src/flowcean/hydra/model.py`
- Test: `tests/hydra/test_hydra_simulation.py`

- [ ] **Step 1: Add failing multi-dimensional and selector tests**

Append this code to `tests/hydra/test_hydra_simulation.py`:

```python
def test_manual_multi_output_hydra_model_simulates_multi_dimensional_state() -> None:
    schema = HyDRATraceSchema(
        time="t",
        state=("x", "v"),
        derivative=("dx", "dv"),
    )
    model = HyDRAModel(
        [LinearDerivativeModel(("dx", "dv"))],
        input_features=["t", "x", "v"],
        output_features=["dx", "dv"],
        trace_schema=schema,
    )

    trace = model.simulate((0.0, 1.0), [0.0, 2.0], sample_dt=0.5)

    assert np.allclose(trace.x[-1], [1.0, 1.5])


class ThresholdSelector:
    def __init__(self) -> None:
        from flowcean.hydra.selector import SelectorFeatureConfig

        self.feature_config = SelectorFeatureConfig(state_columns=("x",))

    def predict_details(self, input_features: pl.DataFrame) -> list[object]:
        from flowcean.hydra.selector import ModePredictionResult

        mode_id = 1 if input_features["x"][0] >= 0.5 else 0
        return [ModePredictionResult(ready=True, mode_id=mode_id)]


class ConstantDerivativeModel(Model):
    def __init__(self, value: float) -> None:
        self.value = value

    def _predict(
        self,
        input_features: pl.LazyFrame | pl.DataFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame({"dx": [self.value] * frame.height}).lazy()


def test_hydra_simulation_selects_mode_once_per_interval() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [ConstantDerivativeModel(1.0), ConstantDerivativeModel(3.0)],
        input_features=["t", "x"],
        output_features=["dx"],
        selector=ThresholdSelector(),  # type: ignore[arg-type]
        trace_schema=schema,
    )

    trace = model.simulate((0.0, 1.0), [0.0], sample_dt=0.5)

    assert np.allclose(trace.x[:, 0], [0.0, 0.5, 2.0])
    assert trace.location.tolist() == ["mode_0", "mode_0", "mode_1"]
```

- [ ] **Step 2: Run selector and multi-dimensional tests**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_manual_multi_output_hydra_model_simulates_multi_dimensional_state tests/hydra/test_hydra_simulation.py::test_hydra_simulation_selects_mode_once_per_interval -v
```

Expected: PASS for both tests.

- [ ] **Step 3: Confirm selector routing remains duck-typed**

Keep `HyDRAModel.selector` typed as `HybridDecisionTreeModel | None`, but do not add implementation checks that require an exact class. `_select_mode_id(...)` should access only the selector API already used by `predict_with_diagnostics(...)`:

```python
        if self.selector.feature_config.max_history > 0:
            message = "HyDRA simulation does not support selector history features yet."
            raise NotImplementedError(message)
        frame = self._build_simulation_frame(t, state, input_stream)
        prediction = self.predict_with_diagnostics(frame)
```

Verify `src/flowcean/hydra/model.py` does not contain an `isinstance(self.selector, ...)` check. No code change is needed when the method already follows the snippet above.

- [ ] **Step 4: Run selector and multi-dimensional tests again**

Run:

```bash
uv run pytest tests/hydra/test_hydra_simulation.py::test_manual_multi_output_hydra_model_simulates_multi_dimensional_state tests/hydra/test_hydra_simulation.py::test_hydra_simulation_selects_mode_once_per_interval -v
```

Expected: PASS for both tests.

- [ ] **Step 5: Checkpoint status**

Run:

```bash
git status --short
```

Expected: shows model and test changes; do not commit unless the user explicitly requests it.

## Task 7: Focused and Full Verification

**Files:**
- Verify: `src/flowcean/hydra/schema.py`
- Verify: `src/flowcean/hydra/model.py`
- Verify: `src/flowcean/hydra/learner.py`
- Verify: `src/flowcean/hydra/simulation.py`
- Verify: `src/flowcean/hydra/__init__.py`
- Verify: `tests/hydra/test_hydra_simulation.py`

- [ ] **Step 1: Run all HyDRA tests**

Run:

```bash
uv run pytest tests/hydra -v
```

Expected: PASS for all HyDRA tests.

- [ ] **Step 2: Run type checks**

Run:

```bash
uv run --all-packages --all-extras basedpyright
```

Expected: PASS with no type errors.

- [ ] **Step 3: Run repository check target**

Run:

```bash
just check
```

Expected: PASS. On failure outside the touched HyDRA files, capture the failure and do not modify unrelated user changes without explicit approval.

- [ ] **Step 4: Final status check**

Run:

```bash
git status --short
```

Expected: only planned files are modified or added. Do not commit unless the user explicitly requests it.

## Plan Self-Review

- Spec coverage: Tasks cover `HyDRATraceSchema`, learner/model schema storage, `predict_next_state`, `simulate`, recursive rollout, comparison, public exports, single-output learner limitation, manual multi-output simulation, selector no-history routing, and verification.
- Placeholder scan: The plan contains concrete paths, tests, code snippets, commands, and expected results. There are no unresolved placeholders.
- Type consistency: The plan consistently uses `HyDRATraceSchema`, `trace_schema`, `predict_next_state`, `simulate`, `StateTraceComparison`, and `compare_state_traces`.
