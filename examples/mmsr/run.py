"""Learn sampled thermostat mappings and source-time guards at a fixed target."""

from itertools import count
from typing import Any

import numpy as np
import polars as pl
from pysr import PySRRegressor

from flowcean.hybrid import simulate
from flowcean.hybrid.benchmarks import thermostat
from flowcean.hybrid.mmsr import fit_mmsr_csr, fit_mmsr_transitions


def main() -> None:
    # Predeclared, identical per-search budgets for CSR and TM. PySR iterations
    # are not Eureqa generations. The dedicated just recipe uses one thread.
    options: dict[str, Any] = {
        "niterations": 120,
        "ncycles_per_iteration": 80,
        "populations": 2,
        "population_size": 32,
        "maxsize": 11,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": [],
        "precision": 64,
        "parallelism": "serial",
        "deterministic": True,
        "parsimony": 0.0,
        "use_frequency": False,
        "use_frequency_in_tournament": False,
        "batching": False,
        "timeout_in_seconds": 45,
        "progress": False,
        "verbosity": 0,
    }
    seeds = count(20260923)
    traces = []
    for initial_temperature in (19.0, 20.0, 20.5, 19.5):
        # Unchanged native physics, fixed target 22. All starts are below the
        # lower switch threshold and share the initial heating regime.
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
            ).select("t", pl.col("x0").alias("temperature"))
        )
    training = [
        trace.with_columns(
            pl.col("temperature").shift(-1).alias("next_temperature")
        ).head(trace.height - 1)
        for trace in traces[:-1]
    ]
    held_out = traces[-1]
    # Drop only the unavailable last target, within each trace. Keep every
    # mixed switch interval; neither event labels nor physical rates are used.
    print(f"Training mapping rows: {sum(t.height for t in training)}")
    print(f"Within-trace TM pairs: {sum(t.height - 1 for t in training)}")
    csr = fit_mmsr_csr(
        training,
        input_features=["temperature"],
        output_feature="next_temperature",
        n_locations=2,
        search_factory=lambda: PySRRegressor(
            **options, random_state=next(seeds)
        ),
        max_sweeps=10,
        random_state=20260920,
    )
    print(f"CSR: {csr.diagnostics}")
    for location, expression in enumerate(csr.expressions):
        print(
            f"Location {location}: next_temperature = {expression}; "
            f"complexity={csr.complexities[location]}, "
            f"variance={csr.variances[location]:.8g}"
        )
    transitions = fit_mmsr_transitions(
        training,
        csr,
        search_factory=lambda: PySRRegressor(
            **options, random_state=next(seeds)
        ),
    )
    for edge in transitions.edges:
        print(
            f"Guard {edge.source_location}->{edge.destination_location}: "
            f"{edge.expression} >= 0; complexity={edge.complexity}, "
            f"weighted MSE={edge.loss:.8g}, AIC={edge.aic:.8g}"
        )
    outgoing = {edge.source_location: edge for edge in transitions.edges}
    # No held-out posterior or native location initializes the predictor.
    location = int(np.argmax(csr.memberships[0][0]))
    predicted = [float(held_out["temperature"][0])]
    for _ in range(held_out.height - 1):
        regressors = pl.DataFrame({"temperature": [predicted[-1]]})
        # Both decisions use OLD predicted temperature and OLD location.
        next_temperature = float(csr.predict(regressors)[0, location])
        edge = outgoing[
            location
        ]  # Exactly one outgoing guard in this K=2 case.
        if edge.guard(regressors)[0]:
            location = edge.destination_location
        predicted.append(next_temperature)
    error = np.asarray(predicted[1:]) - held_out["temperature"].to_numpy()[1:]
    print(f"Recursive held-out RMSE: {np.sqrt(np.mean(error**2)):.8g}")
    print(f"Maximum held-out absolute error: {np.max(np.abs(error)):.8g}")
    print(
        "Fixed-target sampled maps, not ODEs or arbitrary-target response. "
        "Future held-out measurements are used only for scoring."
    )


if __name__ == "__main__":
    main()
