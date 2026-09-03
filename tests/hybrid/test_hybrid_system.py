"""Tests for hybrid-system construction and parameter handling."""

import numpy as np
import pytest

from flowcean.hybrid import (
    CrossingDirection,
    HybridSystem,
    Location,
    SurfaceEntryPolicy,
    Transition,
    simulate,
)


def _zero_flow() -> np.ndarray:
    return np.array([0.0])


def test_hybrid_system_normalizes_sequences_and_copies_parameters() -> None:
    """System collections and parameter mappings are captured at creation."""
    location = Location(_zero_flow)
    parameters = {"gain": 2.0}

    system = HybridSystem(
        locations=[location],
        transitions=[],
        initial_location=location,
        initial_state=np.array([1.0]),
        parameters=parameters,
    )
    parameters["gain"] = 10.0

    assert system.locations == (location,)
    assert system.transitions == ()
    assert system.parameters == {"gain": 2.0}


def test_hybrid_system_rejects_duplicate_location_objects() -> None:
    """A location object can occur only once in a system."""
    location = Location(_zero_flow)

    with pytest.raises(ValueError, match="duplicate Location"):
        HybridSystem(
            [location, location],
            [],
            location,
            np.array([0.0]),
        )


def test_hybrid_system_requires_initial_location_in_locations() -> None:
    """The initial location is validated by identity, not equality."""
    included = Location(_zero_flow)
    external = Location(_zero_flow)

    with pytest.raises(ValueError, match="initial_location"):
        HybridSystem([included], [], external, np.array([0.0]))


def test_hybrid_system_requires_transition_endpoints_in_locations() -> None:
    """Transitions cannot refer to locations outside their system."""
    included = Location(_zero_flow)
    external = Location(_zero_flow)
    transition = Transition(included, external, lambda state: state[0])

    with pytest.raises(ValueError, match="transition target"):
        HybridSystem(
            [included],
            [transition],
            included,
            np.array([0.0]),
        )


def test_transition_entry_policy_defaults_and_requires_exact_enum() -> None:
    """Entry policy is explicit, keyword-only, and strongly typed."""
    source = Location(_zero_flow)
    target = Location(_zero_flow)

    transition = Transition(source, target, lambda: 1.0)

    assert transition.entry_policy is SurfaceEntryPolicy.ERROR
    assert [policy.value for policy in SurfaceEntryPolicy] == [
        "error",
        "trigger",
        "continue",
    ]
    for invalid in ("trigger", CrossingDirection.EITHER, 0):
        with pytest.raises(TypeError, match="SurfaceEntryPolicy"):
            Transition(
                source,
                target,
                lambda: 1.0,
                entry_policy=invalid,  # pyright: ignore[reportArgumentType]
            )
    with pytest.raises(TypeError):
        Transition(
            source,
            target,
            lambda: 1.0,
            None,
            SurfaceEntryPolicy.TRIGGER,  # pyright: ignore[reportCallIssue]
        )


def test_self_transition_without_reset_has_explanatory_note() -> None:
    """A no-op self-transition is rejected with modeling alternatives."""
    location = Location(_zero_flow)

    with pytest.raises(ValueError, match="requires a reset") as caught:
        Transition(location, location, lambda: 1.0)

    notes = " ".join(caught.value.__notes__)
    assert "neither location nor state" in notes
    assert "root localization" in notes
    assert "reset" in notes
    assert "another target" in notes
    assert "remove the transition" in notes
    assert "Flowcean" not in f"{caught.value} {notes}"


def test_location_parameters_override_globals_for_callbacks() -> None:
    """Source-location parameters take precedence in every callback."""
    source = Location(
        lambda parameters: np.array([parameters["rate"]]),
        label="source",
        parameters={"rate": 2.0, "threshold": 1.0, "offset": 3.0},
    )
    target = Location(_zero_flow, label="target")
    transition = Transition(
        source,
        target,
        lambda state, parameters: state[0] - parameters["threshold"],
        lambda state, parameters: np.array(
            [state[0] + parameters["offset"]],
        ),
    )
    system = HybridSystem(
        [source, target],
        [transition],
        source,
        np.array([0.0]),
        parameters={"rate": 20.0, "threshold": 10.0, "offset": 30.0},
    )

    trace = simulate(system, (0.0, 1.0), sample_times=[0.0, 0.5, 1.0])

    assert len(trace.events) == 1
    assert trace.events[0].time == pytest.approx(0.5, abs=1e-7)
    assert trace.events[0].state_after == pytest.approx(np.array([4.0]))
    assert trace.x[1] == pytest.approx(np.array([4.0]))
    assert trace.location.tolist() == ["source", "target", "target"]
