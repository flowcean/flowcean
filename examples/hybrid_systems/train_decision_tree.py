from pathlib import Path
from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import polars as pl
from boiler import Boiler
from jaxtyping import PyTree
from sklearn.tree import DecisionTreeClassifier
from tree_system import Bound, HybridDecisionTree
from typing_extensions import Self, override

import flowcean.cli
import flowcean.utils
from flowcean.core import (
    ChainedOfflineEnvironments,
    Invertible,
    Lambda,
    Transform,
    learn_offline,
)
from flowcean.ode import HybridSystem, evaluate_at, rollout
from flowcean.polars import DataFrame, SlidingWindow
from flowcean.sklearn import DecisionTree, SciKitModel

# -----------------------------
# Configuration and setup
# -----------------------------
config = flowcean.cli.initialize()
flowcean.utils.initialize_random(config.seed)
data_dir = Path("data/boiler/")
dot_graph_export_path = Path("out.dot")


# -----------------------------
# Data preprocessing
# -----------------------------
def extrapolate_mode_time(df: pl.LazyFrame) -> pl.LazyFrame:
    dt = pl.col("t_1") - pl.col("t_0")
    return df.with_columns((pl.col("t_mode_0") + dt).alias("t_mode_1"))


def load_and_preprocess_data(folder: Path) -> DataFrame:
    """Load CSVs, apply sliding window, and compute mode times."""
    chained = ChainedOfflineEnvironments(
        DataFrame.from_csv(path)
        | SlidingWindow(window_size=2)
        | Lambda(extrapolate_mode_time)
        for path in folder.glob("*.csv")
    )
    return DataFrame.concat(chained)


data = load_and_preprocess_data(data_dir)
train = data


class ModeEncoding(Invertible, Transform):
    """Map categorical modes to integers."""

    feature: str
    cat_to_int: dict[str, int] | None = None
    int_to_cat: dict[int, str] | None = None

    def __init__(self, feature: str) -> None:
        super().__init__()
        self.feature = feature

    @override
    def fit(self, data: pl.LazyFrame) -> Self:
        # sort to ensure consistent ordering
        categories = (
            data.select(self.feature).unique().collect().to_series().sort()
        )
        self.cat_to_int = {mode: i for i, mode in enumerate(categories)}
        self.int_to_cat = {i: mode for mode, i in self.cat_to_int.items()}
        return self

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.with_columns(
            pl.col(self.feature).replace(self.cat_to_int).cast(pl.Int64),
        )

    @override
    def inverse(self) -> Transform:
        return Lambda(
            lambda data: data.with_columns(
                pl.col(self.feature).replace(self.int_to_cat),
            ),
        )


# -----------------------------
# Learn decision tree
# -----------------------------
mode_feature = "mode_0"
time_feature = "t_mode_1"
state_features = ["x0_1"]
inputs = [mode_feature, time_feature, *state_features]
outputs = ["mode_1"]

input_transform = ModeEncoding("mode_0")
output_transform = ModeEncoding("mode_1")

learner = DecisionTree(
    ccp_alpha=0.0,
    dot_graph_export_path=dot_graph_export_path,
)
model: SciKitModel[DecisionTreeClassifier] = cast(
    "SciKitModel[DecisionTreeClassifier]",
    learn_offline(
        train,
        learner,
        inputs,
        outputs,
        input_transform=input_transform,
        output_transform=output_transform,
    ),
)

# -----------------------------
# Construct hybrid system
# -----------------------------
# Generic flow functions for any discrete modes in input_transform
flows = {
    "heating": lambda _t, _x, _args: jnp.array(
        [config.boiler.system.heating_rate],
    ),
    "cooling": lambda _t, _x, _args: jnp.array(
        [config.boiler.system.cooling_rate],
    ),
}

hybrid_tree = HybridDecisionTree(
    flows=flows,
    tree=model.estimator,
    input_names=inputs,
    mode_feature=mode_feature,
    mode_decoding=input_transform.int_to_cat or {},
    time_feature=time_feature,
    features=state_features,
)


# -----------------------------
# Pretty-print tree transitions
# -----------------------------
def expression_str(feature: str, bound: Bound) -> str:
    parts = []
    if bound.left is not None:
        parts.append(f"{feature} >= {bound.left:.3f}")
    if bound.right is not None:
        parts.append(f"{feature} < {bound.right:.3f}")
    return " and ".join(parts)


def print_transitions(tree: HybridDecisionTree) -> None:
    for (source, target), conditions in tree.transitions.items():
        if source == target:
            continue
        print(f"{source} -> {target}:")
        for cond in conditions:
            cond_str = " and ".join(
                expression_str(f, b) for f, b in sorted(cond.items())
            )
            print(f"  {cond_str}")


print_transitions(hybrid_tree)


# -----------------------------
# Simulation & comparison
# -----------------------------
def simulate_and_compare(
    tree: HybridDecisionTree,
    system: HybridSystem,
    mode0: str,
    x0: PyTree,
    t0: float,
    t1: float,
    dt0: float,
    dt: float,
) -> None:
    # Hybrid tree simulation
    traces_tree = tree.simulate(mode0=mode0, x0=x0, t0=t0, t1=t1, dt0=dt0)
    data_tree = rollout(traces_tree, dt=dt)

    # Reference system simulation
    traces_sys = system.simulate(mode0=mode0, x0=x0, t0=t0, t1=t1, dt0=dt0)
    data_sys = rollout(traces_sys, dt=dt)

    # Evaluate tree at system timestamps
    data_tree_eval = evaluate_at(
        data_sys.select(pl.col("t")).to_series().to_list(),
        traces_tree,
    )

    # Plot comparison
    _fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    # Line plots with markers
    ax[0].plot(data_sys["t"], data_sys["x0"], marker=".", label="system")
    ax[0].plot(data_tree["t"], data_tree["x0"], marker=".", label="tree sim")
    ax[0].scatter(
        data_tree_eval["t"],
        data_tree_eval["x0"],
        marker="x",
        label="tree eval",
    )
    ax[0].legend()
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("State x0")

    # Error bar plot
    ax[1].bar(
        data_sys["t"],
        data_sys["x0"] - data_tree_eval["x0"],
        width=dt * 0.8,
    )
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Error (system - tree)")
    plt.tight_layout()
    plt.show()


# Example usage
simulate_and_compare(
    tree=hybrid_tree,
    system=Boiler(**config.boiler.system),
    mode0="cooling",
    x0=jnp.array([24.0]),
    t0=0.0,
    t1=5.0,
    dt0=0.01,
    dt=0.1,
)
