from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import override

import numpy as np
import polars as pl
from scipy.integrate import solve_ivp

from flowcean.core.model import Model
from flowcean.hydra.schema import HyDRATraceSchema
from flowcean.hydra.selector import (
    HybridDecisionTreeModel,
    ModePredictionResult,
)
from flowcean.hydra.selector.features import build_selector_inference_frame
from flowcean.ode import InputStream, Trace


@dataclass(frozen=True)
class HyDRABatchPrediction:
    outputs: pl.DataFrame
    row_indices: list[int]
    selector_results: list[ModePredictionResult]


class HyDRAModel(Model):
    """Model composed of learned continuous modes and an optional selector.

    A single-mode model predicts directly with that mode. A multi-mode model
    needs a selector for batch prediction. Model persistence uses Flowcean's
    trusted-only pickle-based model serialization.
    """

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

    def _validate_trace_schema(self) -> None:
        if self.trace_schema is None:
            return
        self.trace_schema.validate_input_features(self.input_features)
        self.trace_schema.validate_output_features(self.output_features)

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        prediction = self.predict_with_diagnostics(input_features)
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        if len(prediction.row_indices) != frame.height:
            message = (
                "HyDRAModel batch prediction requires the stateful "
                "selector runtime "
                "when selector warmup omits rows."
            )
            raise NotImplementedError(message)
        return prediction.outputs.lazy()

    def predict_with_diagnostics(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> HyDRABatchPrediction:
        if not self.modes:
            message = "HyDRAModel contains no learned modes."
            raise ValueError(message)
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        if len(self.modes) == 1:
            return HyDRABatchPrediction(
                outputs=self.modes[0]
                .predict(frame.select(self.input_features))
                .collect()
                .select(self.output_features),
                row_indices=list(range(frame.height)),
                selector_results=[
                    ModePredictionResult(
                        ready=True,
                        mode_id=0,
                        flow_model=self.modes[0],
                    )
                    for _ in range(frame.height)
                ],
            )
        if self.selector is None:
            message = (
                "HyDRAModel prediction requires a mode selector when multiple "
                "modes were learned."
            )
            raise NotImplementedError(message)

        selector_frame = build_selector_inference_frame(
            frame,
            self.selector.feature_config,
        )
        selector_results = self.selector.predict_details(
            selector_frame.features,
        )
        row_indices = selector_frame.row_metadata["row_index"].to_list()

        routed_outputs: list[pl.DataFrame] = []
        predicted_mode_ids = list(
            dict.fromkeys(
                result.mode_id
                for result in selector_results
                if result.mode_id is not None
            ),
        )
        for mode_id in predicted_mode_ids:
            if mode_id < 0 or mode_id >= len(self.modes):
                message = f"selector predicted unknown mode ID {mode_id}"
                raise ValueError(message)

            routed_row_indices = [
                row_index
                for row_index, result in zip(
                    row_indices,
                    selector_results,
                    strict=True,
                )
                if result.mode_id == mode_id
            ]
            if not routed_row_indices:
                continue

            mode_inputs = frame[routed_row_indices].select(self.input_features)
            mode_outputs = (
                self.modes[mode_id]
                .predict(mode_inputs)
                .collect()
                .select(self.output_features)
                .with_columns(pl.Series("__row_index", routed_row_indices))
            )
            routed_outputs.append(mode_outputs)

        outputs = (
            pl.concat(routed_outputs, how="vertical")
            .sort("__row_index")
            .drop("__row_index")
            .select(self.output_features)
            if routed_outputs
            else pl.DataFrame({name: [] for name in self.output_features})
        )

        return HyDRABatchPrediction(
            outputs=outputs,
            row_indices=row_indices,
            selector_results=selector_results,
        )

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
        schema = self._require_trace_schema()
        times = _prepare_simulation_times(t_span, sample_times, sample_dt)
        should_capture_inputs = _should_capture_inputs(
            capture_inputs=capture_inputs,
            input_stream=input_stream,
        )

        state = self._coerce_state(x0, schema)
        states = [state]
        locations: list[str] = []

        for t_start, t_end in pairwise(times):
            frame = self._build_simulation_frame(
                float(t_start),
                state,
                input_stream,
                schema,
            )
            mode_id = self._select_mode_id(frame)
            location = f"mode_{mode_id}"
            if not locations:
                locations.append(location)
            state = self._integrate_next_state(
                self.modes[mode_id],
                state,
                t=float(t_start),
                dt=float(t_end - t_start),
                input_stream=input_stream,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            states.append(state)
            locations.append(location)

        inputs = None
        if should_capture_inputs:
            if input_stream is None:
                message = "Internal error: expected input_stream for capture."
                raise RuntimeError(message)
            inputs = self._capture_simulation_inputs(
                times,
                input_stream,
                schema,
            )

        return Trace(
            t=times,
            x=np.vstack(states),
            location=np.asarray(locations, dtype=object),
            events=(),
            u=inputs,
            dx=None,
        )

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
        initial_state = self._coerce_state(state, schema)
        if not np.isfinite(t):
            message = "t must be finite."
            raise ValueError(message)
        if not np.isfinite(dt) or dt <= 0:
            message = "dt must be finite and greater than zero."
            raise ValueError(message)

        initial_frame = self._build_simulation_frame(
            t,
            initial_state,
            input_stream,
            schema,
        )
        mode_id = self._select_mode_id(initial_frame)
        flow_model = self.modes[mode_id]

        return self._integrate_next_state(
            flow_model,
            initial_state,
            t=t,
            dt=dt,
            input_stream=input_stream,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

    def _integrate_next_state(
        self,
        flow_model: Model,
        state: Iterable[float],
        *,
        t: float,
        dt: float,
        input_stream: InputStream | None,
        rtol: float,
        atol: float,
        max_step: float | None,
    ) -> np.ndarray:
        schema = self._require_trace_schema()
        initial_state = self._coerce_state(state, schema)
        if not np.isfinite(t):
            message = "t must be finite."
            raise ValueError(message)
        if not np.isfinite(dt) or dt <= 0:
            message = "dt must be finite and greater than zero."
            raise ValueError(message)

        def derivative(time: float, current_state: np.ndarray) -> np.ndarray:
            return self._predict_derivative(
                flow_model,
                time,
                current_state,
                input_stream,
                schema,
            )

        solve_kwargs = {
            "fun": derivative,
            "t_span": (t, t + dt),
            "y0": initial_state,
            "t_eval": [t + dt],
            "rtol": rtol,
            "atol": atol,
        }
        if max_step is not None:
            solve_kwargs["max_step"] = max_step

        result = solve_ivp(**solve_kwargs)
        if not result.success:
            message = f"ODE integration failed: {result.message}"
            raise RuntimeError(message)
        return result.y[:, -1]

    def _require_trace_schema(self) -> HyDRATraceSchema:
        if self.trace_schema is None:
            message = "HyDRAModel.predict_next_state requires trace_schema."
            raise ValueError(message)
        self.trace_schema.validate_state_derivative_width()
        return self.trace_schema

    def _coerce_state(
        self,
        state: Iterable[float],
        schema: HyDRATraceSchema,
    ) -> np.ndarray:
        state_array = np.asarray(list(state), dtype=float)
        if state_array.ndim != 1 or state_array.size != len(schema.state):
            message = "state dimension must match trace_schema.state."
            raise ValueError(message)
        if not np.all(np.isfinite(state_array)):
            message = "state values must be finite."
            raise ValueError(message)
        return state_array

    def _select_mode_id(self, frame: pl.DataFrame) -> int:
        if not self.modes:
            message = "HyDRAModel contains no learned modes."
            raise ValueError(message)
        if len(self.modes) == 1:
            return 0
        if self.selector is None:
            message = (
                "HyDRAModel simulation requires a mode selector when multiple "
                "modes were learned."
            )
            raise NotImplementedError(message)
        if self.selector.feature_config.max_history > 0:
            message = "stateful selector simulation is not implemented."
            raise NotImplementedError(message)

        selector_frame = build_selector_inference_frame(
            frame,
            self.selector.feature_config,
        )
        selector_results = self.selector.predict_details(
            selector_frame.features,
        )
        if not selector_results or selector_results[0].mode_id is None:
            message = "selector did not produce a mode for simulation."
            raise ValueError(message)
        mode_id = selector_results[0].mode_id
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
        schema: HyDRATraceSchema,
    ) -> np.ndarray:
        frame = self._build_simulation_frame(t, state, input_stream, schema)
        output_frame = (
            flow_model.predict(frame.select(self.input_features))
            .collect()
            .select(self.output_features)
        )
        derivative = np.asarray(output_frame.row(0), dtype=float)
        if derivative.ndim != 1 or derivative.size != len(schema.derivative):
            message = (
                "Derivative output width must match trace_schema.derivative."
            )
            raise ValueError(message)
        return derivative

    def _build_simulation_frame(
        self,
        t: float,
        state: np.ndarray,
        input_stream: InputStream | None,
        schema: HyDRATraceSchema,
    ) -> pl.DataFrame:
        values = {schema.time: [t]}
        values.update(
            {
                column: [value]
                for column, value in zip(schema.state, state, strict=True)
            },
        )
        values.update(self._simulation_inputs(input_stream, t, schema))
        return pl.DataFrame(values).select(self.input_features)

    def _simulation_inputs(
        self,
        input_stream: InputStream | None,
        t: float,
        schema: HyDRATraceSchema,
    ) -> dict[str, list[float]]:
        if not schema.inputs:
            return {}
        if input_stream is None:
            message = "input_stream is required when trace_schema has inputs."
            raise ValueError(message)
        input_values = np.asarray(input_stream(t), dtype=float)
        if input_values.ndim != 1:
            message = "input_stream must return a 1D array."
            raise ValueError(message)
        if input_values.size != len(schema.inputs):
            message = "input dimension must match trace_schema.inputs."
            raise ValueError(message)
        if not np.all(np.isfinite(input_values)):
            message = "input values must be finite."
            raise ValueError(message)
        return {
            column: [value]
            for column, value in zip(schema.inputs, input_values, strict=True)
        }

    def _capture_simulation_inputs(
        self,
        times: np.ndarray,
        input_stream: InputStream,
        schema: HyDRATraceSchema,
    ) -> np.ndarray:
        values: list[np.ndarray] = []
        input_dim: int | None = None
        for time in times:
            input_values = np.asarray(input_stream(float(time)), dtype=float)
            if input_values.ndim != 1:
                message = "input_stream must return a 1D array."
                raise ValueError(message)
            if schema.inputs and input_values.size != len(schema.inputs):
                message = "input dimension must match trace_schema.inputs."
                raise ValueError(message)
            if input_dim is None:
                input_dim = input_values.size
            elif input_values.size != input_dim:
                message = "Input stream dimension changed during simulation."
                raise ValueError(message)
            if not np.all(np.isfinite(input_values)):
                message = "input values must be finite."
                raise ValueError(message)
            values.append(input_values)

        if not values:
            return np.empty((0, 0), dtype=float)
        return np.vstack(values)


def _prepare_simulation_times(
    t_span: tuple[float, float],
    sample_times: Iterable[float] | None,
    sample_dt: float | None,
) -> np.ndarray:
    t_start = float(t_span[0])
    t_end = float(t_span[1])
    if not np.isfinite(t_start) or not np.isfinite(t_end):
        message = "t_span values must be finite."
        raise ValueError(message)
    if t_end <= t_start:
        message = "t_span end must be greater than start."
        raise ValueError(message)
    if (sample_times is None) == (sample_dt is None):
        message = "Exactly one of sample_times or sample_dt is required."
        raise ValueError(message)

    if sample_times is not None:
        times = _simulation_times_from_samples(sample_times)
    else:
        times = _simulation_times_from_dt(t_start, t_end, sample_dt)

    _validate_simulation_time_grid(times, t_start, t_end)
    return times


def _simulation_times_from_samples(
    sample_times: Iterable[float],
) -> np.ndarray:
    times = np.asarray(list(sample_times), dtype=float)
    if times.size == 0:
        message = "sample_times must be nonempty."
        raise ValueError(message)
    if not np.all(np.isfinite(times)):
        message = "sample_times must be finite."
        raise ValueError(message)
    return times


def _simulation_times_from_dt(
    t_start: float,
    t_end: float,
    sample_dt: float | None,
) -> np.ndarray:
    if sample_dt is None or not np.isfinite(sample_dt):
        message = "sample_dt must be finite."
        raise ValueError(message)
    if sample_dt <= 0:
        message = "sample_dt must be positive."
        raise ValueError(message)
    times = np.arange(t_start, t_end, sample_dt, dtype=float)
    endpoint_atol = float(np.finfo(float).eps * max(1.0, abs(t_end)))
    if times.size and np.isclose(
        times[-1],
        t_end,
        rtol=0.0,
        atol=endpoint_atol,
    ):
        times[-1] = t_end
    else:
        times = np.append(times, t_end)
    return times


def _validate_simulation_time_grid(
    times: np.ndarray,
    t_start: float,
    t_end: float,
) -> None:
    if np.any(np.diff(times) <= 0):
        message = "sample_times must be strictly increasing."
        raise ValueError(message)
    if float(times[0]) != t_start or float(times[-1]) != t_end:
        message = "sample_times must start at t_span[0] and end at t_span[1]."
        raise ValueError(message)


def _should_capture_inputs(
    *,
    capture_inputs: bool | None,
    input_stream: InputStream | None,
) -> bool:
    if capture_inputs is None:
        return input_stream is not None
    if capture_inputs:
        if input_stream is None:
            message = "capture_inputs=True requires an input_stream."
            raise ValueError(message)
        return True
    return False
