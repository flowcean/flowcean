"""Opt-in public PySR integration; ordinary collection never imports Julia."""

import os

import numpy as np
import pytest
from scipy.special import expit

from flowcean.hybrid.mmsr._search import _search


@pytest.mark.skipif(
    os.environ.get("FLOWCEAN_TEST_PYSR") != "1",
    reason="Set FLOWCEAN_TEST_PYSR=1 to run real weighted PySR searches",
)
@pytest.mark.parametrize("objective", ["csr", "transition"])
def test_weighted_frontier_and_numpy_exports(objective):
    from pysr import PySRRegressor

    inputs = np.linspace(-1, 1, 12)[:, None]
    targets = inputs[:, 0] ** 2 if objective == "csr" else expit(inputs[:, 0])
    weights = np.linspace(0.1, 2, len(targets))
    models = []

    def factory():
        model = PySRRegressor(
            niterations=2,
            ncycles_per_iteration=10,
            populations=2,
            population_size=16,
            maxsize=7,
            binary_operators=["+", "*"],
            precision=64,
            parallelism="serial",
            deterministic=True,
            random_state=12,
            batching=False,
            progress=False,
            verbosity=0,
        )
        models.append(model)
        return model

    frontier = _search(
        inputs, targets, weights, ("u",), factory, objective=objective
    )
    assert frontier
    model = models[0]
    assert not os.path.exists(model.output_directory_)
    # Independently verify the backend's weighted loss normalization, not only
    # the library's recomputed loss. Saved exports no longer need the search.
    for _, row in model.equations_.iterrows():
        prediction = row["lambda_format"](inputs)
        residual = (
            prediction - targets
            if objective == "csr"
            else expit(prediction) - targets
        )
        losses = np.log1p(abs(residual)) if objective == "csr" else residual**2
        assert row["loss"] == pytest.approx(
            np.average(losses, weights=weights), abs=1e-10
        )
    for candidate in frontier:
        prediction = candidate.function.predict(inputs)
        residual = (
            prediction - targets
            if objective == "csr"
            else expit(prediction) - targets
        )
        losses = np.log1p(abs(residual)) if objective == "csr" else residual**2
        assert candidate.loss == pytest.approx(
            np.average(losses, weights=weights)
        )
