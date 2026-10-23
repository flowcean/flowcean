"""Compare symbolic guards and hardened-label trees on fixed MMSR dynamics."""

from itertools import count, pairwise
from typing import Any

import numpy as np
import polars as pl
from pysr import PySRRegressor
from sklearn.tree import export_text

from flowcean.hybrid import simulate
from flowcean.hybrid.benchmarks import thermostat
from flowcean.hybrid.famos import fit_famos_transitions
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
    # Stored memberships describe mapping rows, not the final raw sample.
    # np.argmax breaks ties at the lowest positional ID. Never infer a final
    # membership or remove switch-straddling intervals before pairing rows.
    labeled = [
        trace.with_columns(
            pl.Series(
                "location_id",
                [*np.argmax(membership, axis=1).tolist(), None],
                dtype=pl.Int64,
            )
        )
        for trace, membership in zip(traces[:-1], csr.memberships, strict=True)
    ]
    selector = fit_famos_transitions(
        labeled,
        output_features=["temperature"],
        input_features=[],
        n_locations=2,
    )
    tree_pairs = sum(
        sum(a is not None and b is not None for a, b in pairwise(ids))
        for ids in (trace["location_id"].to_list() for trace in labeled)
    )
    print(f"Tree training raw rows: {[trace.height for trace in labeled]}")
    print(f"Within-trace tree pairs: {tree_pairs}")
    print(
        f"Tree outputs: {selector.output_features}, scales={selector.output_scales}; "
        f"inputs: {selector.input_features}, scales={selector.input_scales}"
    )
    for source, tree in enumerate(selector.trees):
        print(f"Next-location tree from {source} (normalized features):")
        print(
            export_text(
                tree,
                feature_names=list(selector.output_features),
                max_depth=tree.get_depth() + 1,
                decimals=8,
            )
        )
    outgoing = {edge.source_location: edge for edge in transitions.edges}
    # No held-out posterior or native location initializes either predictor.
    initial_location = int(np.argmax(csr.memberships[0][0]))
    initial_temperature = float(held_out["temperature"][0])
    print(
        f"Shared CSR dynamics; initial temperature={initial_temperature}, "
        f"initial location={initial_location}. "
        "Soft-membership TM vs hardened-label trees (different training targets)."
    )
    for method in ("MMSR TM", "FaMoS trees"):
        location = initial_location
        predicted = [initial_temperature]
        for _ in range(held_out.height - 1):
            regressors = pl.DataFrame({"temperature": [predicted[-1]]})
            # Both decisions use OLD predicted temperature and OLD location,
            # with the exact same CSR artifact for both recursive evaluations.
            next_temperature = float(csr.predict(regressors)[0, location])
            if method == "MMSR TM":
                # Exactly one outgoing guard per source in this K=2 case.
                edge = outgoing[location]
                next_location = (
                    edge.destination_location
                    if edge.guard(regressors)[0]
                    else location
                )
            else:
                next_location = selector.predict(location, [predicted[-1]], [])
            predicted.append(next_temperature)
            location = next_location
        # Both predictors start from the same supplied temperature seed.
        error = (
            np.asarray(predicted[1:]) - held_out["temperature"].to_numpy()[1:]
        )
        print(f"{method} scored held-out rows: {error.size}/{held_out.height}")
        print(
            f"{method} recursive held-out RMSE: {np.sqrt(np.mean(error**2)):.8g}"
        )
        print(
            f"{method} maximum held-out absolute error: {np.max(np.abs(error)):.8g}"
        )
    print(
        "Fixed-target sampled maps, not ODEs or arbitrary-target response. "
        "Future held-out measurements are used only for scoring."
    )


if __name__ == "__main__":
    main()
