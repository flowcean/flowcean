"""Numerical and staged-data contracts, without running symbolic search."""

from typing import Never

import numpy as np
import polars as pl
import pytest
import sympy as sp

from flowcean.hybrid.mmsr import (
    MMSRCSR,
    CSRDiagnostics,
    fit_mmsr_csr,
    fit_mmsr_transitions,
)
from flowcean.hybrid.mmsr import csr as csr_module
from flowcean.hybrid.mmsr import transitions as tm_module
from flowcean.hybrid.mmsr._search import _aic, _Candidate, _Expression
from flowcean.hybrid.mmsr.csr import (
    _choose_global,
    _responsibilities,
    _state,
    _variance,
)
from flowcean.hybrid.mmsr.transitions import _balanced_weights


def unused_factory() -> Never:
    raise AssertionError("Fixture frontiers must not invoke the backend")


def candidate(expression, complexity=1, loss=0.1):
    return _Candidate(
        _Expression(sp.sympify(expression), complexity, ("u",)), loss
    )


def test_gaussian_prefactors_stability_and_unbiased_variance():
    gamma = _responsibilities(
        np.array([0.0, 1000.0]),
        np.array([[0.0, 0.0], [999.0, 1001.0]]),
        np.array([1.0, 4.0]),
    )
    np.testing.assert_allclose(gamma[0], [2 / 3, 1 / 3])
    density = np.array([np.exp(-0.5), np.exp(-0.125) / 2])
    np.testing.assert_allclose(gamma[1], density / density.sum())
    weights, residual = np.array([0.2, 0.3, 0.5]), np.array([1.0, 2.0, 3.0])
    assert _variance(weights, residual, 1e-12) == pytest.approx(
        np.dot(weights, residual**2) / (1 - np.dot(weights, weights))
    )
    assert _variance(weights, np.zeros(3), 1e-12) == 1e-12
    with pytest.raises(ValueError, match="support"):
        _variance(np.array([1.0, 0.0, 0.0]), residual, 1e-12)


def test_complete_mixture_selection_disagrees_with_local_aic():
    targets = np.tile([1.0, 3.0, 1.0, 1.0, -2.0, 1.0, 1.0, 1.0], 2)
    inputs = np.zeros((len(targets), 1))
    functions = [candidate(-0.8).function, candidate(2.4).function]
    predictions = np.column_stack([f.predict(inputs) for f in functions])
    variances = np.array([0.07, 1.7])
    gamma, _ = _state(targets, predictions, variances)
    # Distinct original trees may simplify to constants. Neither proposal
    # dominates: the better local loss costs more original tree nodes.
    frontier = []
    for value, complexity in [(-0.5, 3), (1.0, 1)]:
        loss = np.average(np.log1p(abs(targets - value)), weights=gamma[:, 0])
        frontier.append(candidate(value, complexity, loss))
    local = min(
        frontier,
        key=lambda c: _aic(c.function.complexity, len(targets), c.loss),
    )
    assert local is frontier[0]
    selected = _choose_global(
        frontier,
        location=0,
        inputs=inputs,
        targets=targets,
        functions=functions,
        predictions=predictions,
        variances=variances,
        variance_floor=1e-12,
    )
    assert selected is not None
    assert selected[0] is frontier[1].function
    temporary = predictions.copy()
    temporary[:, 0] = 1.0
    temporary_gamma = _responsibilities(targets, temporary, variances)
    assert selected[1] == pytest.approx(
        _variance(temporary_gamma[:, 0], targets - 1, 1e-12)
    )


def test_bad_proposals_are_isolated_and_worse_incumbent_replacement_is_allowed():
    targets = np.array([-1.0] * 3 + [1.0] * 3)
    inputs = np.zeros((6, 1))
    functions = [candidate(-1).function, candidate(1).function]
    predictions = np.column_stack([f.predict(inputs) for f in functions])
    variances = np.array([0.1, 0.1])
    _, incumbent_error = _state(targets, predictions, variances)
    replacement = candidate(-0.9)
    chosen = _choose_global(
        [candidate("1/u"), candidate(1000), replacement],
        location=0,
        inputs=inputs,
        targets=targets,
        functions=functions,
        predictions=predictions,
        variances=variances,
        variance_floor=1e-12,
    )
    assert chosen is not None
    assert chosen[0] is replacement.function
    temporary = predictions.copy()
    temporary[:, 0] = replacement.function.predict(inputs)
    gamma = _responsibilities(targets, temporary, variances)
    replacement_error = np.sum(gamma * abs(targets[:, None] - temporary))
    assert _aic(2, 6, replacement_error) > _aic(2, 6, incumbent_error)
    np.testing.assert_array_equal(predictions[:, 0], -1)
    np.testing.assert_array_equal(variances, [0.1, 0.1])


def test_proposal_variance_must_preserve_other_locations_support():
    inputs = np.array([[0.0], [0.0], [10.0], [10.0]])
    targets = inputs[:, 0]
    functions = [
        _Expression(sp.Float("0.1"), 1, ("x",)),
        _Expression(sp.Integer(10), 1, ("x",)),
    ]
    predictions = np.column_stack([f.predict(inputs) for f in functions])
    variances = np.array([0.01, 100.0])
    original_predictions, original_variances = (
        predictions.copy(),
        variances.copy(),
    )
    gamma, _ = _state(targets, predictions, variances)
    np.testing.assert_allclose(gamma.sum(axis=0), [1.98019802, 2.01980198])
    offered_constant = _Candidate(functions[0], np.log1p(0.1))
    near_identity = _Candidate(
        _Expression(sp.Float("1.0000000001") * sp.Symbol("x"), 3, ("x",)),
        0.0,
    )
    selected = _choose_global(
        [offered_constant, near_identity],
        location=0,
        inputs=inputs,
        targets=targets,
        functions=functions,
        predictions=predictions,
        variances=variances,
        variance_floor=1e-12,
    )
    assert selected is not None
    assert selected[0] is offered_constant.function
    # The constant is eligible only because it was explicitly offered. The
    # near-identity proposal's new variance starves the other location.
    assert (
        _choose_global(
            [near_identity],
            location=0,
            inputs=inputs,
            targets=targets,
            functions=functions,
            predictions=predictions,
            variances=variances,
            variance_floor=1e-12,
        )
        is None
    )
    np.testing.assert_array_equal(predictions, original_predictions)
    np.testing.assert_array_equal(variances, original_variances)


def test_unusable_frontier_preserves_valid_csr_and_original_row_alignment(
    monkeypatch,
):
    frontiers = iter([[candidate(-1)], [candidate(1)], [candidate(1000)]])
    monkeypatch.setattr(
        csr_module, "_search", lambda *a, **kw: next(frontiers)
    )
    traces = [
        pl.DataFrame({"u": [0.0, 1.0, 2.0], "y": [-1.0, -1.0, 1.0]}),
        pl.DataFrame({"u": [3.0, 4.0, 5.0], "y": [1.0, 1.0, -1.0]}),
    ]
    model = fit_mmsr_csr(
        traces,
        input_features=["u"],
        output_feature="y",
        n_locations=2,
        search_factory=unused_factory,
        max_sweeps=8,
    )
    assert model.expressions == (-1, 1)
    assert model.diagnostics.stop_reason == "no_usable_candidate"
    assert model.diagnostics.failed_location == 0
    assert model.diagnostics.completed_sweeps == model.diagnostics.updates == 0
    for trace, gamma in zip(traces, model.memberships, strict=True):
        assert gamma.shape == (3, 2)
        np.testing.assert_allclose(gamma, model.membership(trace))
    with pytest.raises(ValueError, match="Missing columns"):
        model.membership(traces[0].select("u"))


def test_error_stability_counts_complete_sweeps_not_discovery(monkeypatch):
    monkeypatch.setattr(csr_module, "_search", lambda *a, **kw: [candidate(0)])
    model = fit_mmsr_csr(
        [
            pl.DataFrame(
                {"u": [0.0, 1.0, 2.0, 3.0], "y": [-1.0, 1.0, -1.0, 1.0]}
            )
        ],
        input_features=["u"],
        output_feature="y",
        n_locations=2,
        search_factory=unused_factory,
        max_sweeps=10,
    )
    assert model.expressions == (0, 0)  # Stable collapse is not discovery.
    assert model.diagnostics.stop_reason == "error_stable"
    assert model.diagnostics.completed_sweeps == 5
    assert model.diagnostics.updates == 10


def test_soft_source_balance_and_trace_local_pairs_for_all_edges(monkeypatch):
    traces = [
        pl.DataFrame({"u": [10.0, 11.0, 12.0]}),
        pl.DataFrame({"u": [20.0, 21.0]}),
    ]
    memberships = (
        np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.1, 0.4, 0.5]]),
        np.array([[0.2, 0.5, 0.3], [0.3, 0.2, 0.5]]),
    )
    csr = MMSRCSR(
        ("u",),
        "y",
        np.ones(3),
        memberships,
        CSRDiagnostics(1, 3, "max_sweeps", 1, 1),
        tuple(candidate(i).function for i in range(3)),
    )
    calls = []

    def search(inputs, targets, weights, *args, **kwargs):
        calls.append((inputs.copy(), targets.copy(), weights.copy()))
        return [candidate("u - 11", 3, 0.2)]

    monkeypatch.setattr(tm_module, "_search", search)
    transitions = fit_mmsr_transitions(
        traces, csr, search_factory=unused_factory
    )
    assert [
        (e.source_location, e.destination_location) for e in transitions.edges
    ] == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    ]
    current = np.vstack([memberships[0][:-1], memberships[1][:-1]])
    following = np.vstack([memberships[0][1:], memberships[1][1:]])
    for edge, (inputs, targets, weights) in zip(
        transitions.edges, calls, strict=True
    ):
        np.testing.assert_array_equal(inputs[:, 0], [10, 11, 20])
        np.testing.assert_array_equal(
            targets, following[:, edge.destination_location]
        )
        k = edge.source_location
        departure = current[:, k] * (1 - following[:, k])
        stay = current[:, k] * following[:, k]
        expected = departure + stay * departure.sum() / stay.sum()
        np.testing.assert_allclose(weights, expected)
        assert (weights - departure).sum() == pytest.approx(departure.sum())
    np.testing.assert_array_equal(calls[0][2], calls[1][2])
    np.testing.assert_array_equal(
        transitions.edges[0].guard(traces[0]), [False, True, True]
    )
    assert transitions.edges[0].expression == sp.Symbol("u") - 11
    with pytest.raises(ValueError, match="departure/stay"):
        _balanced_weights(np.ones(3), np.ones(3))
    monkeypatch.setattr(tm_module, "_search", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="0->1: no usable candidate"):
        fit_mmsr_transitions(traces, csr, search_factory=unused_factory)


def test_expression_constant_broadcast_and_nonfinite_prediction():
    np.testing.assert_array_equal(
        candidate(2).function.predict(np.zeros((4, 1))), [2] * 4
    )
    for value in (sp.nan, sp.oo, sp.zoo):
        with pytest.raises(ValueError, match="nonfinite"):
            candidate(value)
    rational = candidate("1/u").function
    np.testing.assert_allclose(
        rational.predict(np.array([[1.0], [2.0]])), [1, 0.5]
    )
    with pytest.raises(ValueError, match="nonfinite"):
        rational.predict(np.array([[0.0]]))
