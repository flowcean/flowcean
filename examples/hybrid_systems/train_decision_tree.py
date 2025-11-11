from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import polars as pl
from boiler import Boiler
from plot import plot_traces
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
from flowcean.ode import rollout
from flowcean.ode.hybrid_system import evaluate_at
from flowcean.polars import (
    DataFrame,
    SlidingWindow,
    TrainTestSplit,
)
from flowcean.sklearn import DecisionTree, SciKitModel

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier

config = flowcean.cli.initialize()
flowcean.utils.initialize_random(config.seed)


def extrapolate_mode_time(data: pl.LazyFrame) -> pl.LazyFrame:
    dt = pl.col("t_1") - pl.col("t_0")
    return data.with_columns(
        (pl.col("t_mode_0") + dt).alias("t_mode_1"),
    )


data_dir = Path("data/boiler")

data = DataFrame.concat(
    ChainedOfflineEnvironments(
        DataFrame.from_csv(path)
        | SlidingWindow(window_size=2)
        | Lambda(extrapolate_mode_time)
        for path in data_dir.glob("*.csv")
    ),
)

# train, test = TrainTestSplit(ratio=0.05, shuffle=True).split(data)
train = data

train.observe().sink_csv("train.csv")

learner = DecisionTree(ccp_alpha=0.0, dot_graph_export_path="out.dot")


class ModeEncoding(Invertible, Transform):
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


inputs = ["mode_0", "t_mode_1", "x0_1"]
outputs = ["mode_1"]

input_transform = ModeEncoding("mode_0")
output_transform = ModeEncoding("mode_1")
model = learn_offline(
    train,
    learner,
    inputs,
    outputs,
    input_transform=input_transform,
    output_transform=output_transform,
)
model = cast("SciKitModel[DecisionTreeClassifier]", model)

print(input_transform.int_to_cat)
print(output_transform.int_to_cat)

hybrid_decision_tree = HybridDecisionTree(
    flows={
        "heating": lambda _t, _x, _args: jnp.array([10.0]),
        "cooling": lambda _t, _x, _args: jnp.array([-2.0]),
    },
    tree=model.estimator,
    input_names=inputs,
    mode_feature_name="mode_0",
    mode_decoding=input_transform.int_to_cat or {},
)


def expression_str(feature: str, bound: Bound) -> str:
    parts = []
    if bound.left is not None:
        parts.append(f"{feature} >= {bound.left:.3f}")
    if bound.right is not None:
        parts.append(f"{feature} < {bound.right:.3f}")
    return " and ".join(parts)


for (source, target), conditions in hybrid_decision_tree.transitions.items():
    if source == target:
        continue
    print(f"{source} -> {target}:")
    for cond in conditions:
        condition = list(cond.items())
        condition.sort(key=lambda item: item[0])
        cond_str = " and ".join(
            expression_str(feat, b) for feat, b in condition
        )
        print(f"  {cond_str}")

mode0 = "cooling"
x0 = jnp.array([24.0])
t0 = 0.0
t1 = 5.0
dt0 = 0.01
dt = 0.1

traces_tree = hybrid_decision_tree.simulate(
    mode0=mode0,
    x0=x0,
    t0=t0,
    t1=t1,
    dt0=dt0,
)
data_tree = rollout(traces_tree, dt=dt)
data_tree.write_csv("simulated_boiler.csv")


boiler = Boiler(**config.boiler.system)
traces_boiler = boiler.simulate(
    mode0=mode0,
    x0=x0,
    t0=t0,
    t1=t1,
    dt0=dt0,
)
data_boiler = rollout(traces_boiler, dt=dt)

_fig, ax = plt.subplots(2)
ax[0].plot(data_boiler["t"], data_boiler["x0"], marker=".", label="boiler sim")
ax[0].plot(data_tree["t"], data_tree["x0"], marker=".", label="tree sim")
# plot_traces(ax, data_boiler)
# plot_traces(ax, data_tree, marker="x")

data_tree = evaluate_at(
    data_boiler.select(pl.col("t")).to_series().to_list(),
    traces_tree,
)
ax[0].scatter(data_tree["t"], data_tree["x0"], marker="x", label="tree eval")
ax[0].legend()

# error
ax[1].bar(
    data_boiler["t"],
    data_boiler["x0"] - data_tree["x0"],
    width=dt * 0.8,
)
# plot_traces(ax, data_tree, marker="x")
plt.show()
