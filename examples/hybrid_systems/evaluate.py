from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
from boiler import Boiler, BoilerNoTime
from dynamic_hybrid_system import build_modes
from jaxtyping import PyTree
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from stl_mining import (
    STL_BOILER_TRANSITIONS,
    STL_BOILERNOTIME_TRANSITIONS,
    STL_TANK_TRANSITIONS,
)
from tank_system import NTanks
from train_decision_tree import train_hybrid_decision_tree

import flowcean.cli
import flowcean.utils
from flowcean.ode import HybridSystem, evaluate_at, rollout

import random

@dataclass
class Evaluation:
    reference_rollout: pl.DataFrame
    model_rollout: dict[str, pl.DataFrame]
    model_evaluation: dict[str, pl.DataFrame]


def evaluate(
    reference: HybridSystem,
    models: dict[str, HybridSystem],
    mode0: str,
    x0: PyTree,
    t0: float,
    t1: float,
    dt0: float,
    dt: float,
    init_modes: [str],
    state_max = 1,
    n_runs: int = 1
) -> Evaluation:
    evals = []
    for i in range(n_runs):
        mode_init = random.choice(init_modes)
        state_init = jnp.array(random.sample(range(state_max*10), len(x0)))/10
        # mode_init = mode0
        x0 = x0
        traces_ref = reference.simulate(mode0=mode_init, x0=state_init, t0=t0, t1=t1, dt0=dt0)
        reference_rollout = rollout(traces_ref, dt=dt)
        model_rollout = {
            name: rollout(
                system.simulate(
                    mode0=mode_init,
                    x0=state_init,
                    t0=t0,
                    t1=t1,
                    dt0=dt0,
                ),
                dt=dt,
            )
            for name, system in models.items()
        }
        ts = reference_rollout.select(pl.col("t")).to_series().to_list()
        model_evaluation = {
            name: evaluate_at(
                ts,
                system.simulate(
                    mode0=mode_init,
                    x0=state_init,
                    t0=t0,
                    t1=t1,
                    dt0=dt0,
                ),
            )
            for name, system in models.items()
        }
        evals.append(Evaluation(
            reference_rollout=reference_rollout,
            model_rollout=model_rollout,
            model_evaluation=model_evaluation,
            )
        )
    return evals

def plot_evaluation_traces(ax: Axes, evaluation: Evaluation) -> None:
    cmap = plt.get_cmap("tab10")
    model_colors = {
        name: cmap(i) for i, name in enumerate(evaluation.model_rollout)
    }

    state_features = evaluation.reference_rollout.select(
        cs.starts_with("x"),
    ).columns

    for state_feature in state_features:
        ax.plot(
            evaluation.reference_rollout["t"],
            evaluation.reference_rollout[state_feature],
            marker=".",
            label=f"system {state_feature}",
            color="black",
        )
        for name, trace in evaluation.model_rollout.items():
            ax.plot(
                trace["t"],
                trace[state_feature],
                linestyle="--",
                label=f"{name} {state_feature}",
                color=model_colors[name],
            )
        for name, trace in evaluation.model_evaluation.items():
            ax.scatter(
                trace["t"],
                trace[state_feature],
                marker="x",
                label=f"{name} eval {state_feature}",
                color=model_colors[name],
            )
    legend_handles = []

    # Reference system
    legend_handles.append(
        Line2D(
            [],
            [],
            color="black",
            marker=".",
            linestyle="-",
            label="system",
        ),
    )

    # Each model
    for name, color in model_colors.items():
        legend_handles.append(
            Line2D(
                [],
                [],
                color=color,
                linestyle="--",
                label=f"{name} rollout",
            ),
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker="x",
                linestyle="None",
                label=f"{name} eval",
            ),
        )

    ax.legend(handles=legend_handles)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")


# n_models = len(models)
# _fig, ax = plt.subplots(
#     1,  # n_models + 1,
#     1,
#     figsize=(10, 6),
#     layout="constrained",
# )
# ax = [ax]

# Error bar plot
# for i, (name, data_tree_eval) in enumerate(data_models_eval.items()):
#     ax[i + 1].set_title(f"Error: system - {name}")
#     for state_feature in state_features:
#         ax[i + 1].bar(
#             data_ref["t"],
#             data_ref[state_feature] - data_tree_eval[state_feature],
#             width=dt * 0.8,
#             label=f"{state_feature}",
#         )
#     ax[i + 1].legend()
#     ax[i + 1].set_xlabel("Time")
#     ax[i + 1].set_ylabel("Error (system - tree)")

config = flowcean.cli.initialize()
flowcean.utils.initialize_random(config.seed)

boiler_flows = {
    "heating": lambda _t, _x, _args: jnp.array(
        [config.boiler.system.heating_rate],
    ),
    "cooling": lambda _t, _x, _args: jnp.array(
        [config.boiler.system.cooling_rate],
    ),
}

leakages = jnp.array(config.tank.system.leakages)
inflows = jnp.array(config.tank.system.inflows)
tank_flows = {
    "all_leak": lambda _t, x, _args: leakages * x,
    "flow_0": lambda _t, x, _args: leakages * x
    + jnp.array([inflows[0], 0.0, 0.0]),
    "flow_1": lambda _t, x, _args: leakages * x
    + jnp.array([0.0, inflows[1], 0.0]),
    "flow_2": lambda _t, x, _args: leakages * x
    + jnp.array([0.0, 0.0, inflows[2]]),
}

stl_boiler = HybridSystem(
    build_modes(
        flows=boiler_flows,
        transitions=STL_BOILER_TRANSITIONS,
        time_feature="t",
        features=["x0"],
    ),
)

stl_boilernotime = HybridSystem(
    build_modes(
        flows=boiler_flows,
        transitions=STL_BOILERNOTIME_TRANSITIONS,
        time_feature="t",
        features=["x0"],
    ),
)

stl_tank = HybridSystem(
    build_modes(
        flows=tank_flows,
        transitions=STL_TANK_TRANSITIONS,
        time_feature="t",
        features=["x0", "x1", "x2"],
    ),
)

hdt_boiler = train_hybrid_decision_tree(
    data_dir="data/boiler",
    input_mode_feature="mode_0",
    output_mode_feature="mode_1",
    time_feature="t_mode_1",
    state_features=["x0_1"],
    flows=boiler_flows,
)
hdt_boiler.print_transitions()

hdt_boilernotime = train_hybrid_decision_tree(
    data_dir="data/boilernotime",
    input_mode_feature="mode_0",
    output_mode_feature="mode_1",
    time_feature="t_mode_1",
    state_features=["x0_1"],
    flows=boiler_flows,
)
hdt_boilernotime.print_transitions()

hdt_tank = train_hybrid_decision_tree(
    data_dir="data/tank",
    input_mode_feature="mode_0",
    output_mode_feature="mode_1",
    time_feature="t_mode_1",
    state_features=["x0_1", "x1_1", "x2_1"],
    flows=tank_flows,
)
hdt_tank.print_transitions()

evaluations = {
    "boiler": evaluate(
        reference=Boiler(**config.boiler.system),
        models={
            "STL": stl_boiler,
            "HDT": hdt_boiler,
        },
        mode0="cooling",
        x0=jnp.array([23.0]),
        t0=0.0,
        t1=2.0,
        dt0=0.01,
        dt=0.1,
        n_runs = 100,
        state_max = 30,
        init_modes = ["heating", "cooling"]
    ),
    "boilernotime": evaluate(
        reference=BoilerNoTime(**config.boilernotime.system),
        models={
            "STL": stl_boilernotime,
            "HDT": hdt_boilernotime,
        },
        mode0="heating",
        x0=jnp.array([6.5]),
        t0=0.0,
        t1=2.0,
        dt0=0.01,
        dt=0.1,
        n_runs = 100,
        state_max = 30,
        init_modes = ["heating", "cooling"]
    ),
    "tank": evaluate(
        reference=NTanks(**config.tank.system),
        models={
            "STL": stl_tank,
            "HDT": hdt_tank,
        },
        mode0="flow_0",
        x0=jnp.array([0.5, 10.0, 1.5]),
        t0=0.0,
        t1=2.0,
        dt0=0.01,
        dt=0.1,
        n_runs = 100,
        state_max=15,
        init_modes=["flow_0", "flow_1", "flow_2"],
    ),
}

# one evaluation per system
for name, evaluation in evaluations.items():
    n_columns = len(evaluation[0].reference_rollout.drop(['mode','t_mode','t']).columns)
    means = { method :pl.DataFrame(data =[[0]*n_columns]*len(evaluation), schema= list(zip(evaluation[0].reference_rollout.drop(['mode','t_mode','t']).columns,[pl.Float64]*n_columns)),orient='row')
             for method in evaluation[0].model_evaluation.keys()
             }

    for i, run in enumerate(evaluation):
        reference = run.reference_rollout
        ref = reference.drop(['mode','t_mode','t'])
    
        for j,(method, result) in enumerate(run.model_evaluation.items()):
            diff = (ref - result.drop('mode','t')).drop_nans()
            diff *= diff
            mean = diff.mean()
            for c in mean.columns:
                means[method][i, c] = mean[c]
        if i%50 ==0:
            fig, ax = plt.subplots(
                1,
                1,
                layout="constrained",
            )
            plot_evaluation_traces(ax, run)
            ax.set_title(f"Evaluation {i} traces for {name} system")
    print(means)
    for method in means.keys():
        print(method)
        print(means[method].describe())

plt.show()
