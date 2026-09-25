"""Public callback protocols describe the full named call contract."""

from inspect import Parameter, signature

import numpy as np

from flowcean.hybrid import (
    EventSurfaceFunction,
    FlowFunction,
    ResetFunction,
)
from flowcean.hybrid.hybrid_system import InputStream, Parameters, State


def test_callback_protocols_accept_five_named_inputs() -> None:
    def flow(
        *,
        t: float,
        state: State,
        parameters: Parameters,
        input_stream: InputStream,
        location_time: float,
    ) -> State:
        return state + t + parameters["gain"] + input_stream(t) + location_time

    def surface(
        *,
        t: float,
        state: State,
        parameters: Parameters,
        input_stream: InputStream,
        location_time: float,
    ) -> float:
        return float(
            state[0]
            + t
            + parameters["gain"]
            + input_stream(t)[0]
            + location_time
        )

    def reset(
        *,
        t: float,
        state: State,
        parameters: Parameters,
        input_stream: InputStream,
        location_time: float,
    ) -> State:
        return state + t + parameters["gain"] + input_stream(t) + location_time

    flow_callback: FlowFunction = flow
    surface_callback: EventSurfaceFunction = surface
    reset_callback: ResetFunction = reset
    state: State = np.array([2.0])
    parameters: Parameters = {"gain": 3.0}

    def input_stream(t: float) -> State:
        return np.array([t])

    np.testing.assert_allclose(
        flow_callback(
            t=1.0,
            state=state,
            parameters=parameters,
            input_stream=input_stream,
            location_time=4.0,
        ),
        [11.0],
    )
    assert (
        surface_callback(
            t=1.0,
            state=state,
            parameters=parameters,
            input_stream=input_stream,
            location_time=4.0,
        )
        == 11.0
    )
    np.testing.assert_allclose(
        reset_callback(
            t=1.0,
            state=state,
            parameters=parameters,
            input_stream=input_stream,
            location_time=4.0,
        ),
        [11.0],
    )


def test_callback_protocol_signatures_are_five_required_keywords() -> None:
    canonical = ("t", "state", "parameters", "input_stream", "location_time")
    for protocol in (FlowFunction, EventSurfaceFunction, ResetFunction):
        parameters = signature(protocol.__call__).parameters
        assert tuple(parameters) == ("self", *canonical)
        for name in canonical:
            assert parameters[name].kind is Parameter.KEYWORD_ONLY
            assert parameters[name].default is Parameter.empty
