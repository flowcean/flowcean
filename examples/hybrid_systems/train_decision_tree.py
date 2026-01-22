from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl
from hybrid_decision_trees import HybridDecisionTree
from typing_extensions import Self, override

from flowcean.core import (
    ChainedOfflineEnvironments,
    Invertible,
    Lambda,
    Transform,
    learn_offline,
)
from flowcean.ode.hybrid_system import FlowFn
from flowcean.polars import DataFrame, SlidingWindow
from flowcean.sklearn import DecisionTree, SciKitModel

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier


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


def train_hybrid_decision_tree(
    data_dir: str | PathLike[str],
    input_mode_feature: str,
    output_mode_feature: str,
    time_feature: str,
    state_features: Sequence[str],
    flows: dict[str, FlowFn],
    dot_graph_export_path: Path | None = None,
) -> HybridDecisionTree:
    data = load_and_preprocess_data(Path(data_dir))

    input_transform = ModeEncoding(input_mode_feature)
    output_transform = ModeEncoding(output_mode_feature)

    learner = DecisionTree(
        ccp_alpha=0.0,
        dot_graph_export_path=dot_graph_export_path,
    )

    inputs = [input_mode_feature, time_feature, *state_features]
    outputs = [output_mode_feature]

    model: SciKitModel[DecisionTreeClassifier] = cast(
        "SciKitModel[DecisionTreeClassifier]",
        learn_offline(
            data,
            learner,
            inputs,
            outputs,
            input_transform=input_transform,
            output_transform=output_transform,
        ),
    )

    return HybridDecisionTree(
        flows=flows,
        tree=model.estimator,
        input_names=inputs,
        mode_feature=input_mode_feature,
        mode_decoding=input_transform.int_to_cat or {},
        time_feature=time_feature,
        features=state_features,
    )


# stl_system = HybridSystem(
#     build_modes(
#         flows=flows,
#         transitions=STL_TANK_TRANSITIONS,
#         time_feature="t",
#         features=["x0", "x1", "x2"],
#     ),
# )
#
# # Example usage
# simulate_and_compare(
#     reference=NTanks(**config.tank.system),
#     models={"hybrid_tree": hybrid_tree, "stl_system": stl_system},
#     state_features=["x0", "x1", "x2"],
#     mode0="all_leak",
#     x0=jnp.array([15.0, 15.0, 15.0]),
#     t0=0.0,
#     t1=5.0,
#     dt0=0.01,
#     dt=0.1,
# )
