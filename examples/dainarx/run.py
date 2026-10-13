"""Identify a small sampled switching system, then recursively predict it."""

import numpy as np
import polars as pl

from flowcean.hybrid.dainarx import fit_dainarx


def sampled_trace(seed: int) -> pl.DataFrame:
    """Generate independent noise-free observations, not training labels.

    Input ``u`` commands +2/-2 switch locations on the following row; ordinary
    inputs lie in [-0.5, 0.5]. A switch changes both dynamics and output ``x``.
    Every trace starts in the same location, but with a different output.

    Args:
        seed: Random seed for inputs and the initial output.

    Returns:
        A Polars frame with ``time`` (sample timestamps), ``u`` (input), and
            ``x`` (output) columns.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.5, 0.5, 240)
    for index, t in enumerate(range(25, len(u), 25)):
        u[t] = 2 if index % 2 == 0 else -2
    x = np.empty(len(u))
    x[0] = rng.uniform(-1, 1)
    location = 0
    for t in range(1, len(u)):
        switch = (location == 0 and u[t - 1] > 1) or (
            location == 1 and u[t - 1] < -1
        )
        if switch:
            location = 1 - location
            x[t] = 0.4 * x[t - 1] + 0.1 * u[t] + (1 if location else -1)
        elif location == 0:
            x[t] = 0.8 * x[t - 1] + 0.2 * u[t] + 0.1
        else:
            x[t] = -0.5 * x[t - 1] + 0.3 * u[t] - 0.2
    return pl.DataFrame({"time": np.arange(len(u)) * 0.1, "u": u, "x": x})


def main() -> None:
    training = [sampled_trace(seed) for seed in range(4)]
    fit = fit_dainarx(
        training,
        input_features=["u"],
        output_features=["x"],
        time_column="time",
        order=1,
        window=6,
        guard_kernel="linear",
        guard_c=1e4,
        guard_gamma="scale",
    )
    held_out = sampled_trace(42)
    # All traces start in the same physical location. Its learned ID is known
    # from the first training segment, not inferred from future test outputs.
    initial_location = fit.segments[0].location_id
    predicted = fit.model.rollout(
        held_out.select("time", "u"),
        output_history=held_out.select("x").head(1).to_numpy(),
        initial_location=initial_location,
    )
    error = predicted["x"].to_numpy()[1:] - held_out["x"].to_numpy()[1:]
    print(
        f"{len(fit.segments)} segments; {len(fit.model.locations)} locations"
    )
    print(f"Recursive held-out RMSE: {np.sqrt(np.mean(error**2)):.3g}")
    assert np.max(np.abs(error)) < 1e-10


if __name__ == "__main__":
    main()
