"""Identify sampled Duffing dynamics with Dainarx."""

import numpy as np
import polars as pl

from flowcean.hybrid import simulate
from flowcean.hybrid.benchmarks import duffing_oscillator
from flowcean.hybrid.dainarx import (
    DainarxModel,
    fit_dainarx_dynamics,
    fit_dainarx_transitions,
)


def main() -> None:
    order = 2
    traces = []
    for initial_state in (
        (1.2, 0.0),
        (1.1, 0.15),
        (1.3, -0.1),
        (1.0, -0.2),
        (1.15, -0.075),  # Held-out trace.
    ):
        trajectory = simulate(
            duffing_oscillator(initial_state=np.asarray(initial_state)),
            t_span=(0.0, 20.0),
            input_stream=lambda t: np.array([np.cos(t)]),
            rtol=1e-11,
            atol=1e-13,
            max_step=0.005,
        )
        traces.append(
            trajectory.sample(dt=0.001, include_inputs=True).select(
                "t",
                pl.col("u0").alias("force"),
                pl.col("x0").alias("position"),
            )
        )
    training, held_out = traces[:-1], traces[-1]
    dynamics = fit_dainarx_dynamics(
        training,
        input_features=["force"],
        output_features=["position"],
        time_column="t",
        order=order,
        window=10,
        nonlinear_terms=(
            lambda history, _: history[-1] ** 3,
            lambda history, _: history[-2] ** 3,
        ),
    )
    print(f"Discovered segments: {len(dynamics.segments)}")
    print(f"Learned locations: {len(dynamics.locations)}")
    print(f"Training RMSE threshold: {dynamics.eta:.6g}")
    for location_id, location_model in enumerate(dynamics.locations):
        print(
            f"Location {location_id} training RMSE: {location_model.rmse:.6g}"
        )

    transitions = fit_dainarx_transitions(
        training,
        dynamics,
        guard_kernel="rbf",
        guard_c=1e6,
        guard_gamma="scale",
    )
    model = DainarxModel(dynamics.locations, transitions)
    # All chosen initial states share a location, before any switch.
    # Its learned ID comes from the first training segment.
    predicted = model.rollout(
        held_out.select("t", "force"),
        output_history=held_out.select("position").head(order).to_numpy(),
        initial_location=dynamics.segments[0].location_id,
    )
    # Exclude the measured position history supplied to the rollout.
    error = (
        predicted["position"].to_numpy()[order:]
        - held_out["position"].to_numpy()[order:]
    )
    print(f"Scored held-out rows: {error.size}/{held_out.height}")
    print(f"Recursive held-out RMSE: {np.sqrt(np.mean(error**2)):.6g}")
    print(f"Maximum held-out absolute error: {np.max(np.abs(error)):.6g}")


if __name__ == "__main__":
    main()
