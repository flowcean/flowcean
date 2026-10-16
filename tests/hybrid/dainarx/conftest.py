import numpy as np
import polars as pl
import pytest

from flowcean.hybrid.dainarx import fit_dainarx


def sampled_system(seed, length=200):
    """Return ``(frame, location_labels)`` for an independent sampled system.

    The frame contains ``time``, output ``x``, and input ``u``. Input commands
    switch between two locations; two successive post-switch predictor phases
    generate outputs before the destination dynamics resume.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.5, 0.5, length)
    for index, t in enumerate(range(20, length, 20)):
        u[t] = 2 if index % 2 == 0 else -2
    x = np.empty(length)
    x[:2] = rng.uniform(-1, 1, 2)
    labels = np.zeros(length, dtype=int)
    location, phase = 0, 2
    for t in range(2, length):
        if t % 20 == 1:
            location = 1 - location
            phase = 0
        sign = 1 if location == 1 else -1
        if phase == 0:
            x[t] = 0.7 * x[t - 1] + 0.2 * x[t - 2] + 0.1 * u[t] + sign
        elif phase == 1:
            x[t] = 0.4 * x[t - 1] - 0.1 * x[t - 2] + 0.1 * u[t] + 0.3 * sign
        elif location == 0:
            x[t] = 0.5 * x[t - 1] - 0.1 * x[t - 2] + 0.2 * u[t] + 0.1
        else:
            x[t] = -0.3 * x[t - 1] + 0.15 * x[t - 2] + 0.3 * u[t] - 0.2
        labels[t] = location
        phase = min(2, phase + 1)
    return pl.DataFrame(
        {"time": np.arange(length) * 0.1, "x": x, "u": u}
    ), labels


@pytest.fixture(scope="module")
def identified():
    training = [sampled_system(seed) for seed in range(4)]
    fit = fit_dainarx(
        [frame for frame, _ in training],
        input_features=["u"],
        output_features=["x"],
        time_column="time",
        order=2,
        window=8,
        guard_kernel="linear",
        guard_c=1e4,
        guard_gamma="scale",
    )
    return fit, training
