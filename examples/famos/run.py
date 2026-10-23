"""Identify sampled thermostat dynamics with FaMoS-DTW at a fixed setpoint."""

import numpy as np
import polars as pl

from flowcean.hybrid import simulate
from flowcean.hybrid.benchmarks import thermostat
from flowcean.hybrid.famos import (
    FamosModel,
    discover_famos_locations,
    fit_famos_dynamics,
    fit_famos_transitions,
)


def main() -> None:
    traces = []
    for initial_temperature in (19.0, 20.0, 20.5, 19.5):
        # The unchanged native factory starts in heating; all starts are below
        # the lower threshold. Continuous events are NOT snapped to samples.
        trajectory = simulate(
            thermostat(initial_state=np.array([initial_temperature])),
            t_span=(0.0, 15.0),
            input_stream=lambda _t: np.array([22.0]),
        )
        traces.append(
            trajectory.sample(
                dt=0.01,
                include_inputs=True,
                include_location_id=False,
                include_location_time=False,
            ).select(
                "t",
                pl.col("u0").alias("target"),
                pl.col("x0").alias("temperature"),
            )
        )
    training, held_out = traces[:-1], traces[-1]
    discovery = discover_famos_locations(
        training,
        input_features=["target"],
        output_features=["temperature"],
        window=10,
        trim=2,
        max_derivative=1,
        clustering_derivative=1,
        peak_height=0.05,
        alpha=2.0,
        rho_min=0.001,
        rho_max=0.03,
    )
    print(f"Discovered segments: {len(discovery.segments)}")
    print(f"Learned locations: {len(discovery.location_keys)}")
    print(
        f"Unassigned rows: {sum(segment.stop - segment.start for segment in discovery.segments if segment.location_id is None)}"
    )
    dynamics = fit_famos_dynamics(training, discovery, order=1)
    selector = fit_famos_transitions(
        discovery.to_labeled_frames(training),
        output_features=discovery.output_features,
        input_features=discovery.input_features,
        n_locations=len(discovery.location_keys),
    )
    for location, error in enumerate(dynamics.rmse):
        print(
            f"Location {location}: n={dynamics.sample_counts[location]}, "
            f"rank={dynamics.ranks[location]}/{dynamics.coefficients[location].shape[0]}, "
            f"training RMSE={error[0]:.6g}"
        )
    # Use a training assignment, not held-out clustering or simulator labels.
    initial_location = next(
        segment.location_id
        for segment in discovery.segments
        if segment.location_id is not None
    )
    predicted = FamosModel(dynamics, selector).rollout(
        held_out.select("t", "target"),
        output_history=held_out.select("temperature").head(1).to_numpy(),
        initial_location=initial_location,
    )
    # The first temperature was supplied as the rollout seed, not predicted.
    error = (
        predicted["temperature"].to_numpy()[1:]
        - held_out["temperature"].to_numpy()[1:]
    )
    print(f"Scored held-out rows: {error.size}/{held_out.height}")
    print(f"Recursive held-out RMSE: {np.sqrt(np.mean(error**2)):.6g}")
    print(f"Maximum held-out absolute error: {np.max(np.abs(error)):.6g}")
    print(
        "Fixed target only: constant input and bias are collinear; arbitrary target response is not identified."
    )


if __name__ == "__main__":
    main()
