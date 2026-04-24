import polars as pl
import pytest

from flowcean.hydra.selector import (
    HybridDecisionTreeLearner,
    SelectorFeatureConfig,
    StatefulHybridDecisionTreeSelector,
    evaluate_selector_autoregressive,
    evaluate_selector_oracle,
)


def test_stateful_selector_returns_not_ready_until_history_and_seed_are_available(  # noqa: E501
) -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), mode_history=1),
        random_state=7,
        max_depth=2,
    )
    model = learner.learn_from_traces(
        [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})],
    )
    runtime = StatefulHybridDecisionTreeSelector(model, seed_modes=[0])

    first = runtime.predict({"x": 0.0})

    assert first.ready is True
    assert first.mode_id == 0


def test_stateful_selector_without_seed_stays_not_ready() -> None:
    learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), mode_history=1),
        random_state=7,
        max_depth=2,
    )
    model = learner.learn_from_traces(
        [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})],
    )
    runtime = StatefulHybridDecisionTreeSelector(model)

    first = runtime.predict({"x": 0.0})
    second = runtime.predict({"x": 0.1})

    assert first.ready is False
    assert first.mode_id is None
    assert second.ready is False
    assert second.mode_id is None


def test_evaluation_helpers_report_oracle_and_autoregressive_accuracy() -> (
    None
):
    config = SelectorFeatureConfig(state_columns=("x",), mode_history=1)
    learner = HybridDecisionTreeLearner(config, random_state=7, max_depth=2)
    traces = [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})]
    model = learner.learn_from_traces(traces)

    oracle = evaluate_selector_oracle(model, traces)
    autoregressive = evaluate_selector_autoregressive(
        model,
        traces,
        seed_modes=[0],
    )

    assert oracle.samples == 3
    assert autoregressive.samples == 4
    assert 0.0 <= oracle.accuracy <= 1.0
    assert 0.0 <= autoregressive.accuracy <= 1.0
    assert oracle.confusion_matrix.height > 0
    assert autoregressive.confusion_matrix.height > 0


def test_autoregressive_evaluation_bootstraps_mode_history_by_default() -> (
    None
):
    config = SelectorFeatureConfig(state_columns=("x",), mode_history=1)
    learner = HybridDecisionTreeLearner(config, random_state=7, max_depth=2)
    traces = [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})]
    model = learner.learn_from_traces(traces)

    report = evaluate_selector_autoregressive(model, traces)

    assert report.samples == 3
    assert 0.0 <= report.accuracy <= 1.0
    assert report.confusion_matrix.height > 0


def test_autoregressive_evaluation_skips_bootstrapped_prefix() -> None:
    config = SelectorFeatureConfig(state_columns=("x",), mode_history=1)
    learner = HybridDecisionTreeLearner(config, random_state=7, max_depth=2)
    traces = [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})]
    model = learner.learn_from_traces(traces)

    implicit_report = evaluate_selector_autoregressive(model, traces)
    explicit_report = evaluate_selector_autoregressive(
        model,
        traces,
        seed_modes=[0],
    )

    assert implicit_report.samples == 3
    assert explicit_report.samples == 4


def test_autoregressive_evaluation_rejects_explicit_short_seed_history() -> (
    None
):
    config = SelectorFeatureConfig(state_columns=("x",), mode_history=2)
    learner = HybridDecisionTreeLearner(config, random_state=7, max_depth=2)
    traces = [
        pl.DataFrame(
            {"x": [0.0, 0.1, 0.2, 1.0, 1.1], "mode": [0, 0, 0, 1, 1]},
        ),
    ]
    model = learner.learn_from_traces(traces)

    with pytest.raises(
        ValueError,
        match="explicit seed history is shorter than configured mode_history",
    ):
        evaluate_selector_autoregressive(model, traces, seed_modes=[0])


def test_autoregressive_evaluation_rejects_traces_without_mode_column() -> (
    None
):
    config = SelectorFeatureConfig(state_columns=("x",), mode_history=1)
    learner = HybridDecisionTreeLearner(config, random_state=7, max_depth=2)
    training_traces = [
        pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]}),
    ]
    model = learner.learn_from_traces(training_traces)

    with pytest.raises(
        ValueError,
        match="selector traces must include a mode column",
    ):
        evaluate_selector_autoregressive(
            model,
            [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1]})],
        )
