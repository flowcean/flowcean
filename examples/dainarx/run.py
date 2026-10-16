"""Identify sampled Duffing dynamics and report incomplete identification honestly."""

import numpy as np
import polars as pl

from flowcean.hybrid import simulate
from flowcean.hybrid.benchmarks import duffing_force, duffing_oscillator
from flowcean.hybrid.dainarx import (
    DainarxModel,
    fit_dainarx_dynamics,
    fit_dainarx_transitions,
)

TIME_SPAN = (0.0, 20.0)
SAMPLING_PERIOD = 0.001
ORDER = 2
TRAINING_INITIAL_STATES = (
    (1.2, 0.0),
    (1.1, 0.15),
    (1.3, -0.1),
    (1.0, -0.2),
)
HELD_OUT_INITIAL_STATE = (1.15, -0.075)


def observe(initial_state: tuple[float, float]) -> pl.DataFrame:
    """Simulate from position and velocity, retaining only learner observations."""
    trajectory = simulate(
        duffing_oscillator(initial_state=np.asarray(initial_state)),
        t_span=TIME_SPAN,
        input_stream=duffing_force,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.005,
    )
    times = np.linspace(
        *TIME_SPAN,
        round((TIME_SPAN[1] - TIME_SPAN[0]) / SAMPLING_PERIOD) + 1,
    )
    return trajectory.sample(times=times, include_inputs=True).select(
        "t",
        pl.col("u0").alias("force"),
        pl.col("x0").alias("position"),
    )


def main() -> None:
    training = [observe(initial) for initial in TRAINING_INITIAL_STATES]
    dynamics = fit_dainarx_dynamics(
        training,
        input_features=["force"],
        output_features=["position"],
        time_column="t",
        order=ORDER,
        window=10,
        nonlinear_terms=(
            lambda history, _: history[-1] ** 3,
            lambda history, _: history[-2] ** 3,
        ),
    )
    print(f"Discovered segments: {len(dynamics.segments)}")
    print(f"Learned locations: {len(dynamics.locations)}")
    print(f"Training RMSE threshold: {dynamics.eta:.6g}")
    for location_id, model in enumerate(dynamics.locations):
        print(f"Location {location_id} training RMSE: {model.rmse:.6g}")

    try:
        transitions = fit_dainarx_transitions(
            training,
            dynamics,
            guard_kernel="rbf",
            guard_c=1e6,
            guard_gamma="scale",
        )
    except ValueError as error:
        print(f"Transition fitting did not complete: {error}")
        print("No executable model or held-out prediction was produced.")
        return

    model = DainarxModel(dynamics.locations, transitions)
    held_out = observe(HELD_OUT_INITIAL_STATE)
    # These initial conditions all start in the same physical location,
    # before any switch. Its learned ID comes from the first training segment.
    try:
        predicted = model.rollout(
            held_out.select("t", "force"),
            output_history=held_out.select("position").head(ORDER).to_numpy(),
            initial_location=dynamics.segments[0].location_id,
        )
    except ValueError as error:
        print(f"Held-out rollout did not complete: {error}")
        return
    error = (
        predicted["position"].to_numpy()[ORDER:]
        - held_out["position"].to_numpy()[ORDER:]
    )
    print(f"Recursive held-out RMSE: {np.sqrt(np.mean(error**2)):.6g}")
    print(f"Maximum held-out absolute error: {np.max(np.abs(error)):.6g}")


if __name__ == "__main__":
    main()
