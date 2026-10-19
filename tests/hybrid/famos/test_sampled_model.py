"""Sample timing, trace isolation, categorical dispatch and recursive rollout."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from sklearn.tree import DecisionTreeClassifier

from flowcean.hybrid.famos import (
    FamosDiscovery,
    FamosDynamics,
    FamosModel,
    FamosSegment,
    FamosTransitionSelector,
    fit_famos_dynamics,
    fit_famos_transitions,
)


def discovery_for(traces, segments, locations=1):
    return FamosDiscovery(
        segments=tuple(FamosSegment(*segment) for segment in segments),
        trace_lengths=tuple(trace.height for trace in traces),
        input_features=("u",),
        output_features=("y",),
        time_column="t",
        dt=1.0,
        input_scales=np.array([2.0]),
        output_scales=np.array([5.0]),
        location_keys=tuple((i,) for i in range(locations)),
    )


def test_arx_source_inputs_boundary_targets_and_trace_local_history():
    traces = []
    for initial in (1.0, -7.0):
        u = np.array([0, 1, -1, 3, 0, -2, 4, 1, 2, -1, 5, 0], dtype=float)
        y = [initial, initial + 2]
        for t in range(1, len(u) - 1):
            y.append(
                100.0 if t == 3 else 2 * y[-1] - 0.5 * y[-2] + 3 * u[t] + 1
            )
        traces.append(
            pl.DataFrame({"t": np.arange(len(u), dtype=float), "u": u, "y": y})
        )
    # Equal labels do not erase the cut: the jump target at row 4 is excluded.
    discovery = discovery_for(
        traces, [(i, a, b, 0) for i in range(2) for a, b in [(0, 4), (4, 12)]]
    )
    dynamics = fit_famos_dynamics(traces, discovery, order=2)
    assert dynamics.sample_counts == (18,)
    assert dynamics.ranks == (4,)
    np.testing.assert_allclose(dynamics.rmse[0], [0], atol=1e-10)
    # Out-of-training-range prediction must not clip or refit stored scales.
    assert dynamics.predict(0, [[1000], [2000]], [20])[0] == pytest.approx(
        3561
    )
    np.testing.assert_array_equal(discovery.output_scales, [5])
    with pytest.raises(ValueError, match="exactly order"):
        dynamics.predict(0, [[1]], [0])


def test_rank_deficiency_reported_and_unusable_discovery_stays_independent():
    trace = pl.DataFrame(
        {"t": np.arange(6, dtype=float), "y": [1.0] * 6, "u": [2.0] * 6}
    )
    discovery = discovery_for([trace], [(0, 0, 6, 0)])
    dynamics = fit_famos_dynamics([trace], discovery)
    assert dynamics.ranks == (1,)
    np.testing.assert_allclose(dynamics.predict(0, [[1]], [2]), [1])
    partial = replace(
        discovery,
        segments=(FamosSegment(0, 0, 5, None), FamosSegment(0, 5, 6, 0)),
    )
    assert (
        partial.to_labeled_frames([trace])[0]["location_id"].null_count() == 5
    )
    with pytest.raises(ValueError, match="no usable regression targets"):
        fit_famos_dynamics([trace], partial)
    with pytest.raises(ValueError, match="no outgoing training pairs"):
        fit_famos_transitions([trace], partial)


def test_next_location_trees_are_source_conditional_and_label_permutation_invariant():
    trace = pl.DataFrame(
        {
            "t": np.arange(8, dtype=float),
            "y": [0, 1, 2, 0, 1, 2, 0, 1],
            "u": [0] * 8,
            "location_id": [99] * 8,  # Unselected native labels are ignored.
        }
    )
    discovery = discovery_for(
        [trace], [(0, 0, 3, 0), (0, 3, 6, 1), (0, 6, 8, 0)], locations=2
    )
    selector = fit_famos_transitions([trace], discovery)
    swapped = replace(
        discovery,
        segments=tuple(
            replace(
                s,
                location_id=1 - s.location_id
                if s.location_id is not None
                else None,
            )
            for s in discovery.segments
        ),
    )
    other = fit_famos_transitions([trace], swapped)
    for source in range(2):
        for y in (0, 1, 2):
            expected = 1 - source if y == 2 else source
            assert selector.predict(source, [y], [0]) == expected
            assert other.predict(1 - source, [y], [0]) == 1 - expected


def test_rollout_uses_old_location_source_input_and_updates_history():
    inputs = pl.DataFrame(
        {
            "t": np.arange(5, dtype=float),
            "u": [99, 2, 3, 4, 999],
            "y": [-100] * 5,
        }
    )
    discovery = discovery_for([inputs], [(0, 0, 5, 0)], locations=2)
    discovery = replace(
        discovery, input_scales=np.ones(1), output_scales=np.ones(1)
    )
    # q0: y_next = y_current + 10*y_previous + u_current.
    # q1: y_next = y_current - y_previous + 2*u_current.
    dynamics = FamosDynamics(
        discovery,
        2,
        (np.array([[1], [10], [1], [0]]), np.array([[1], [-1], [2], [0]])),
        (0, 0),
        (0, 0),
        (np.zeros(1), np.zeros(1)),
    )
    # Switching depends on the OLD output: q0 switches only below y=5.
    tree0 = DecisionTreeClassifier().fit([[0, 0], [10, 0]], [1, 0])
    tree1 = DecisionTreeClassifier().fit([[0, 0]], [0])
    model = FamosModel(
        dynamics, FamosTransitionSelector(discovery, (tree0, tree1))
    )
    result = model.rollout(
        inputs, output_history=[[1], [2]], initial_location=0
    )
    np.testing.assert_array_equal(result["y"], [1, 2, 14, 18, 162])
    assert result["location_id"].to_list() == [None, 0, 1, 0, 0]
    # Final input is unused; unrelated future truth is ignored.
    changed = inputs.with_columns(
        pl.Series("u", [99, 2, 3, 4, -1000]), pl.lit(123).alias("y")
    )
    assert model.rollout(
        changed, output_history=[[1], [2]], initial_location=0
    ).equals(result)
    assert (
        model.rollout(
            inputs.head(2), output_history=[[1], [2]], initial_location=0
        ).height
        == 2
    )
