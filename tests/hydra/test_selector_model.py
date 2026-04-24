import importlib.util
import subprocess
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from flowcean.core import Model
from flowcean.hydra.selector import (
    HybridDecisionTreeLearner,
    HybridDecisionTreeModel,
    ModePredictionResult,
    SelectorFeatureConfig,
    SelectorInspection,
)
from flowcean.hydra.selector import graph as selector_graph
from flowcean.hydra.selector.graph import build_selector_dot
from flowcean.hydra.selector.inspection import (
    SelectorLeafInspection,
    SelectorNodeInspection,
)


class ConstantFlow(Model):
    def __init__(self, value: float, output_name: str = "y") -> None:
        self.value = value
        self.output_name = output_name

    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame(
            {self.output_name: [self.value] * frame.height},
        ).lazy()


class SummaryFlow(ConstantFlow):
    def __init__(self, value: float, summary: str) -> None:
        super().__init__(value)
        self.summary = summary

    def flow_summary(self) -> str:
        return self.summary


def test_hybrid_decision_tree_learner_trains_from_labeled_traces() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=2,
        random_state=7,
    )
    traces = [
        pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]}),
    ]

    model = learner.learn_from_traces(traces)

    assert isinstance(model, HybridDecisionTreeModel)
    assert model.predict(pl.DataFrame({"x": [0.05, 1.05]}).lazy()).collect()[
        "mode"
    ].to_list() == [0, 1]


def test_hybrid_decision_tree_model_returns_diagnostics() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=2,
        random_state=7,
    )
    traces = [
        pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]}),
    ]
    mode_to_flow = cast(
        "dict[int, Model]",
        {0: ConstantFlow(10.0), 1: ConstantFlow(20.0)},
    )

    model = learner.learn_from_traces(traces, mode_to_flow=mode_to_flow)
    result = model.predict_details(pl.DataFrame({"x": [0.05]}))[0]

    assert isinstance(result, ModePredictionResult)
    assert result.ready is True
    assert result.mode_id == 0
    assert result.flow_model is mode_to_flow[0]
    assert result.leaf_id is not None
    assert sum(result.probabilities.values()) == 1.0


def test_hybrid_decision_tree_model_exposes_feature_importances() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), input_columns=("u",)),
        max_depth=2,
        random_state=7,
    )
    traces = [
        pl.DataFrame(
            {
                "x": [0.0, 0.1, 1.0, 1.1],
                "u": [0.0, 0.0, 1.0, 1.0],
                "mode": [0, 0, 1, 1],
            },
        ),
    ]

    model = learner.learn_from_traces(traces)

    assert set(model.feature_importances()) == {"x", "u"}
    assert model.tree_text()


def test_hybrid_decision_tree_model_inspect_returns_structured_payload() -> (
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), input_columns=("u",)),
        max_depth=2,
        random_state=7,
    )
    traces = [
        pl.DataFrame(
            {
                "x": [0.0, 0.1, 1.0, 1.1],
                "u": [0.0, 0.0, 1.0, 1.0],
                "mode": [0, 0, 1, 1],
            },
        ),
    ]

    model = learner.learn_from_traces(traces)
    inspection = model.inspect()

    assert isinstance(inspection, SelectorInspection)
    assert inspection.feature_columns == ("x", "u")
    assert inspection.classes == (0, 1)
    assert inspection.n_leaves >= 2
    assert inspection.max_depth >= 1
    assert inspection.nodes[0].sample_count >= 1
    assert inspection.nodes[0].weighted_class_support
    assert {leaf.mode_id for leaf in inspection.leaves} == {0, 1}
    assert all(leaf.weighted_class_support for leaf in inspection.leaves)
    assert {mode.mode_id for mode in inspection.modes} == {0, 1}
    assert all(mode.weighted_support >= 1 for mode in inspection.modes)


def test_hybrid_decision_tree_model_inspect_reconstructs_leaf_class_counts() -> (  # noqa: E501
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=1,
        random_state=7,
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        pl.DataFrame({"mode": [0, 0, 1, 0]}),
    )
    inspection = model.inspect()

    assert inspection.nodes[0].sample_count == 4
    assert inspection.nodes[0].weighted_class_support == {0: 3, 1: 1}
    assert {leaf.sample_count for leaf in inspection.leaves} == {2}
    assert {
        tuple(
            leaf.weighted_class_support[mode_id]
            for mode_id in inspection.classes
        )
        for leaf in inspection.leaves
    } == {
        (2, 0),
        (1, 1),
    }
    assert {
        mode.mode_id: mode.weighted_support for mode in inspection.modes
    } == {
        0: 3,
        1: 1,
    }


def test_hybrid_decision_tree_model_inspect_uses_weighted_class_support() -> (
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=1,
        random_state=7,
        class_weight={0: 1, 1: 4},
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        pl.DataFrame({"mode": [0, 0, 1, 0]}),
    )
    inspection = model.inspect()

    assert inspection.nodes[0].sample_count == 4
    assert inspection.nodes[0].weighted_class_support == {0: 3.0, 1: 4.0}
    assert {
        mode.mode_id: mode.weighted_support for mode in inspection.modes
    } == {
        0: 3.0,
        1: 4.0,
    }
    assert any(
        leaf.weighted_class_support[1] > leaf.sample_count
        for leaf in inspection.leaves
    )


def test_weighted_selector_summaries_distinguish_samples_from_support() -> (
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=1,
        random_state=7,
        class_weight={0: 1, 1: 4},
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        pl.DataFrame({"mode": [0, 0, 1, 0]}),
    )

    leaf_summary = model.leaf_summary_text()
    mode_summary = model.mode_summary_text()

    assert "raw_samples=2" in leaf_summary
    assert "weighted_support=[2.0, 0.0]" not in leaf_summary
    assert "weighted_class_support=[0=1.0, 1=4.0]" in leaf_summary
    assert "weighted_support=3.0" in mode_summary
    assert "weighted_support=4.0" in mode_summary
    assert "samples=3.0" not in mode_summary


def test_hybrid_decision_tree_model_summary_text_includes_mode_and_leaf_counts() -> (  # noqa: E501
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), input_columns=("u",)),
        max_depth=2,
        random_state=7,
    )
    model = learner.learn_from_traces(
        [
            pl.DataFrame(
                {
                    "x": [0.0, 0.1, 1.0, 1.1],
                    "u": [0.0, 0.0, 1.0, 1.0],
                    "mode": [0, 0, 1, 1],
                },
            ),
        ],
    )

    summary = model.summary_text()

    assert "modes: 2" in summary
    assert "leaves:" in summary
    assert "feature columns:" in summary
    assert "mode summary:" not in summary
    assert "leaf summary:" not in summary


def test_hybrid_decision_tree_model_debug_prediction_text_includes_inputs() -> (  # noqa: E501
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), input_columns=("u",)),
        max_depth=2,
        random_state=7,
    )
    model = learner.learn_from_traces(
        [
            pl.DataFrame(
                {
                    "x": [0.0, 0.1, 1.0, 1.1],
                    "u": [0.0, 0.0, 1.0, 1.0],
                    "mode": [0, 0, 1, 1],
                },
            ),
        ],
        mode_to_flow=cast(
            "dict[int, Model]",
            {0: ConstantFlow(10.0), 1: ConstantFlow(20.0)},
        ),
    )

    debug_text = model.debug_prediction_text(
        pl.DataFrame({"x": [0.05], "u": [0.0]}),
    )

    assert "inputs:" in debug_text
    assert "x=0.05" in debug_text
    assert "u=0.0" in debug_text
    assert "mode=0" in debug_text
    assert "leaf_id=" in debug_text


def test_hybrid_decision_tree_model_leaf_summary_text_formats_multiline_flow_summary() -> (  # noqa: E501
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), input_columns=("u",)),
        max_depth=2,
        random_state=7,
    )
    model = learner.learn_from_traces(
        [
            pl.DataFrame(
                {
                    "x": [0.0, 0.1, 1.0, 1.1],
                    "u": [0.0, 0.0, 1.0, 1.0],
                    "mode": [0, 0, 1, 1],
                },
            ),
        ],
        mode_to_flow=cast(
            "dict[int, Model]",
            {
                0: SummaryFlow(10.0, "line one\nline two"),
                1: SummaryFlow(20.0, "single line"),
            },
        ),
    )

    summary = model.leaf_summary_text()

    assert "leaf_id=" in summary
    assert "flow:\n  line one\n  line two" in summary


def test_hybrid_decision_tree_model_mode_summary_text_formats_multiline_flow_summary() -> (  # noqa: E501
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), input_columns=("u",)),
        max_depth=2,
        random_state=7,
    )
    model = learner.learn_from_traces(
        [
            pl.DataFrame(
                {
                    "x": [0.0, 0.1, 1.0, 1.1],
                    "u": [0.0, 0.0, 1.0, 1.0],
                    "mode": [0, 0, 1, 1],
                },
            ),
        ],
        mode_to_flow=cast(
            "dict[int, Model]",
            {
                0: SummaryFlow(10.0, "alpha\nbeta"),
                1: SummaryFlow(20.0, "gamma"),
            },
        ),
    )

    summary = model.mode_summary_text()

    assert "mode=0" in summary
    assert "flow:\n  alpha\n  beta" in summary


def test_hybrid_decision_tree_model_to_dot_includes_split_and_leaf_content() -> (  # noqa: E501
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), input_columns=("u",)),
        max_depth=2,
        random_state=7,
    )
    model = learner.learn_from_traces(
        [
            pl.DataFrame(
                {
                    "x": [0.0, 0.1, 1.0, 1.1],
                    "u": [0.0, 0.0, 1.0, 1.0],
                    "mode": [0, 0, 1, 1],
                },
            ),
        ],
        mode_to_flow=cast(
            "dict[int, Model]",
            {
                0: SummaryFlow(10.0, "line one"),
                1: SummaryFlow(20.0, "line two"),
            },
        ),
    )

    dot = model.to_dot()

    assert dot.startswith("digraph")
    assert "x <=" in dot or "u <=" in dot
    assert "raw_samples=" in dot
    assert "mode=0" in dot
    assert "flow=line one" in dot or "flow=line two" in dot


def test_hybrid_decision_tree_model_to_svg_returns_svg_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=1,
        random_state=7,
    )
    model = learner.learn_from_traces(
        [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})],
    )

    monkeypatch.setattr(
        selector_graph.shutil,
        "which",
        lambda _cmd: "/usr/bin/dot",
    )
    monkeypatch.setattr(
        selector_graph.subprocess,
        "run",
        lambda *args, **_kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="<svg>tree</svg>",
            stderr="",
        ),
    )

    assert model.to_svg() == "<svg>tree</svg>"


def test_hybrid_decision_tree_model_save_svg_writes_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=1,
        random_state=7,
    )
    model = learner.learn_from_traces(
        [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})],
    )

    monkeypatch.setattr(
        selector_graph.shutil,
        "which",
        lambda _cmd: "/usr/bin/dot",
    )
    monkeypatch.setattr(
        selector_graph.subprocess,
        "run",
        lambda *args, **_kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="<svg>saved</svg>",
            stderr="",
        ),
    )

    output_path = tmp_path / "selector.svg"
    model.save_svg(output_path)

    assert output_path.read_text(encoding="utf-8") == "<svg>saved</svg>"


def test_passive_circuit_example_writes_selector_svg_when_selector_exists(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "passive_circuit_run",
        Path("examples/passive_circuit/run.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FailingSelector:
        def __init__(self) -> None:
            self.saved_path: Path | None = None

        def summary_text(self) -> str:
            return "summary"

        def mode_summary_text(self) -> str:
            return "mode-summary"

        def leaf_summary_text(self) -> str:
            return "leaf-summary"

        def tree_text(self) -> str:
            return "tree"

        def save_svg(self, path: Path) -> None:
            self.saved_path = path
            message = "graphviz missing"
            raise RuntimeError(message)

    selector = FailingSelector()

    module.print_selector_outputs(selector, output_dir=tmp_path)

    assert selector.saved_path == tmp_path / "selector_tree.svg"

    output = capsys.readouterr().out
    assert "selector_summary" in output
    assert "summary" in output
    assert "selector_mode_summary" in output
    assert "mode-summary" in output
    assert "selector_leaf_summary" in output
    assert "leaf-summary" in output
    assert "selector_tree" in output
    assert "tree" in output
    assert "selector_svg_skipped" in output
    assert "graphviz missing" in output


def test_passive_circuit_example_keeps_text_output_when_svg_write_fails(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "passive_circuit_run",
        Path("examples/passive_circuit/run.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FailingSelector:
        def __init__(self) -> None:
            self.saved_path: Path | None = None

        def summary_text(self) -> str:
            return "summary"

        def mode_summary_text(self) -> str:
            return "mode-summary"

        def leaf_summary_text(self) -> str:
            return "leaf-summary"

        def tree_text(self) -> str:
            return "tree"

        def save_svg(self, path: Path) -> None:
            self.saved_path = path
            message = "disk full"
            raise OSError(message)

    selector = FailingSelector()

    module.print_selector_outputs(selector, output_dir=tmp_path)

    assert selector.saved_path == tmp_path / "selector_tree.svg"

    output = capsys.readouterr().out
    assert "selector_summary" in output
    assert "summary" in output
    assert "selector_mode_summary" in output
    assert "mode-summary" in output
    assert "selector_leaf_summary" in output
    assert "leaf-summary" in output
    assert "selector_tree" in output
    assert "tree" in output
    assert "selector_svg_skipped" in output
    assert "disk full" in output


def test_hybrid_decision_tree_model_to_svg_errors_when_graphviz_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=1,
        random_state=7,
    )
    model = learner.learn_from_traces(
        [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})],
    )

    monkeypatch.setattr(selector_graph.shutil, "which", lambda _cmd: None)

    with pytest.raises(RuntimeError, match="install Graphviz"):
        model.to_svg()


def test_hybrid_decision_tree_model_to_dot_preserves_graphviz_line_break_escapes() -> (  # noqa: E501
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), input_columns=("u",)),
        max_depth=2,
        random_state=7,
    )
    model = learner.learn_from_traces(
        [
            pl.DataFrame(
                {
                    "x": [0.0, 0.1, 1.0, 1.1],
                    "u": [0.0, 0.0, 1.0, 1.0],
                    "mode": [0, 0, 1, 1],
                },
            ),
        ],
        mode_to_flow=cast(
            "dict[int, Model]",
            {
                0: SummaryFlow(10.0, "line one"),
                1: SummaryFlow(20.0, "line two"),
            },
        ),
    )

    dot = model.to_dot()

    assert "\\nraw_samples=" in dot
    assert "\\\\nraw_samples=" not in dot
    assert "\\nflow=" in dot


def test_build_selector_dot_raises_clear_error_for_malformed_payload() -> None:
    inspection = SelectorInspection(
        feature_columns=("x",),
        classes=(0, 1),
        max_depth=1,
        n_leaves=1,
        nodes=(
            SelectorNodeInspection(
                node_id=0,
                sample_count=4,
                impurity=0.5,
                is_leaf=False,
                predicted_mode_id=0,
                weighted_class_support={0: 2.0, 1: 2.0},
                feature_index=0,
                feature_name="x",
                threshold=0.5,
                left_child_id=1,
                right_child_id=99,
            ),
            SelectorNodeInspection(
                node_id=1,
                sample_count=2,
                impurity=0.0,
                is_leaf=True,
                predicted_mode_id=0,
                weighted_class_support={0: 2.0, 1: 0.0},
            ),
        ),
        leaves=(
            SelectorLeafInspection(
                node_id=1,
                mode_id=0,
                sample_count=2,
                weighted_class_support={0: 2.0, 1: 0.0},
                flow_summary="line one",
            ),
        ),
        modes=(),
    )

    with pytest.raises(ValueError, match="invalid selector inspection"):
        build_selector_dot(inspection)


def test_selector_leaf_summary_text_includes_flow_function_summary() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=2,
        random_state=7,
    )
    traces = [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})]
    mode_to_flow = cast(
        "dict[int, Model]",
        {0: ConstantFlow(10.0), 1: ConstantFlow(20.0)},
    )

    model = learner.learn_from_traces(traces, mode_to_flow=mode_to_flow)

    leaf_summary = model.leaf_summary_text()

    assert "flow=" in leaf_summary
    assert "ConstantFlow" in leaf_summary


def test_reusing_learner_does_not_mutate_earlier_models() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=1,
        random_state=7,
    )

    first_model = learner.learn(
        pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1]}),
        pl.DataFrame({"mode": [0, 0, 1, 1]}),
    )
    first_prediction_before = (
        first_model.predict(pl.DataFrame({"x": [0.05, 1.05]}))
        .collect()["mode"]
        .to_list()
    )

    second_model = learner.learn(
        pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1]}),
        pl.DataFrame({"mode": [1, 1, 0, 0]}),
    )

    assert first_prediction_before == [0, 1]
    assert first_model.predict(pl.DataFrame({"x": [0.05, 1.05]})).collect()[
        "mode"
    ].to_list() == [0, 1]
    assert second_model.predict(pl.DataFrame({"x": [0.05, 1.05]})).collect()[
        "mode"
    ].to_list() == [1, 0]


def test_hybrid_decision_tree_model_predicts_from_history_features() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), state_history=1),
        max_depth=2,
        random_state=7,
    )

    model = learner.learn(
        pl.DataFrame(
            {
                "x": [0.0, 0.0, 1.0, 1.0],
                "x_t_minus_1": [0.0, 1.0, 0.0, 1.0],
            },
        ),
        pl.DataFrame({"mode": [0, 1, 0, 1]}),
    )

    predictions = model.predict(
        pl.DataFrame(
            {
                "x": [0.0, 1.0],
                "x_t_minus_1": [0.0, 1.0],
            },
        ),
    ).collect()

    assert predictions["mode"].to_list() == [0, 1]


def test_hybrid_decision_tree_model_supports_mode_history_input() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), mode_history=1),
        max_depth=2,
        random_state=7,
    )
    traces = [
        pl.DataFrame(
            {
                "x": [0.0, 0.1, 1.0, 1.1],
                "mode": [0, 0, 1, 1],
            },
        ),
    ]

    model = learner.learn_from_traces(traces)
    predictions = model.predict(
        pl.DataFrame(
            {
                "x": [0.1, 1.0, 1.1],
                "mode_t_minus_1": [0, 0, 1],
            },
        ),
    ).collect()

    assert model.feature_columns == ("x", "mode_t_minus_1")
    assert predictions["mode"].to_list() == [0, 1, 1]


def test_hybrid_decision_tree_model_handles_empty_engineered_input() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=2,
        random_state=7,
    )
    model = learner.learn(
        pl.DataFrame({"x": [0.0, 1.0]}),
        pl.DataFrame({"mode": [0, 1]}),
    )

    predictions = model.predict(
        pl.DataFrame({"x": []}, schema={"x": pl.Float64}),
    )

    assert predictions.collect().to_dict(as_series=False) == {"mode": []}
    assert (
        model.predict_details(
            pl.DataFrame({"x": []}, schema={"x": pl.Float64}),
        )
        == []
    )


def test_hybrid_decision_tree_learner_allows_single_mode_labels() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=2,
        random_state=7,
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 0.1, 0.2]}),
        pl.DataFrame({"mode": [0, 0, 0]}),
    )

    assert model.predict(pl.DataFrame({"x": [0.05, 1.05]})).collect()[
        "mode"
    ].to_list() == [0, 0]


def test_hybrid_decision_tree_learner_trains_from_single_mode_traces() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=2,
        random_state=7,
    )

    model = learner.learn_from_traces(
        [pl.DataFrame({"x": [0.0, 0.1, 0.2], "mode": [0, 0, 0]})],
    )

    details = model.predict_details(pl.DataFrame({"x": [0.05]}))[0]

    assert details.mode_id == 0
    assert details.probabilities == {0: 1.0}


def test_hybrid_decision_tree_learner_rejects_non_integer_mode_labels() -> (
    None
):
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        max_depth=2,
        random_state=7,
    )

    with pytest.raises(ValueError, match="integer-like mode IDs"):
        learner.learn(
            pl.DataFrame({"x": [0.0, 1.0]}),
            pl.DataFrame({"mode": ["a", "b"]}),
        )
