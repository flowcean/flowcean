import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from flowcean.hydra.replay import HyDRAReplayEmitter
from tests.hydra.test_hydra import AlwaysZeroLearner


def test_hydra_replay_emitter_records_steps_and_final_replay() -> None:
    emitter = HyDRAReplayEmitter()

    emitter.start(
        trace_count=1,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )

    emitter.pending_segment_found(0, 0, 3)
    emitter.candidate_window_evaluated(
        0,
        0,
        1,
        window_size=2,
        fit=0.0,
    )
    result = emitter.finish(final_mode_count=1)
    replay = emitter.replay

    assert result is None
    assert replay is not None
    assert [step.kind for step in emitter.steps] == [
        "pending_segment_found",
        "candidate_window_evaluated",
    ]
    assert replay.trace_count == 1
    assert replay.final_mode_count == 1
    assert replay.steps[1].fit == 0.0
    assert not any(
        isinstance(value, pl.DataFrame)
        for step in replay.steps
        for value in step.__dict__.values()
    )


def test_hydra_replay_emitter_is_terminal_after_finish() -> None:
    emitter = HyDRAReplayEmitter()

    emitter.start(
        trace_count=1,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )

    result = emitter.finish(final_mode_count=1)

    assert result is None
    assert emitter.replay is not None
    assert emitter.replay.final_mode_count == 1
    with pytest.raises(RuntimeError, match="already finished"):
        emitter.pending_segment_found(0, 0, 3)
    with pytest.raises(RuntimeError, match="already finished"):
        emitter.finish(final_mode_count=1)


def test_hydra_replay_emitter_exposes_start_not_reset() -> None:
    emitter = HyDRAReplayEmitter()

    assert hasattr(emitter, "start")
    assert not hasattr(emitter, "reset")


def test_hydra_package_does_not_export_internal_callback_protocol() -> None:
    from flowcean import hydra

    assert "HyDRACallback" not in hydra.__all__
    assert not hasattr(hydra, "HyDRACallback")


def _patch_live_plot_subplots(
    monkeypatch: pytest.MonkeyPatch,
) -> list[None]:
    subplots_calls: list[None] = []

    fake_axes = SimpleNamespace(
        clear=lambda: None,
        plot=lambda *_args, **_kwargs: None,
        axvspan=lambda *_args, **_kwargs: None,
        set_title=lambda _value: None,
        set_xlabel=lambda _value: None,
        legend=lambda **_kwargs: None,
    )
    fake_figure = SimpleNamespace(
        canvas=SimpleNamespace(draw=lambda: None),
        subplots_adjust=lambda **_kwargs: None,
    )

    def fake_subplots() -> tuple[object, object]:
        subplots_calls.append(None)
        return (fake_figure, fake_axes)

    def fake_show(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr("flowcean.hydra.live_plot.plt.subplots", fake_subplots)
    monkeypatch.setattr("flowcean.hydra.live_plot.plt.show", fake_show)
    return subplots_calls


def test_hydra_learner_accepts_live_plot_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flowcean.hydra import HyDRALearner
    from flowcean.hydra.live_plot import HyDRALivePlotCallback

    subplots_calls = _patch_live_plot_subplots(monkeypatch)

    callback = HyDRALivePlotCallback(
        traces=[pl.DataFrame({"x": [0.0, 0.1], "y": [0.0, 0.0]})],
        y_columns=("y",),
        x_column="x",
    )
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=1,
        callback=callback,
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 0.1]}).lazy(),
        pl.DataFrame({"y": [0.0, 0.0]}).lazy(),
    )

    assert model is not None
    assert len(subplots_calls) == 1


def test_hydra_learner_uses_start_and_self_callback_directly() -> None:
    from flowcean.hydra.learner import HyDRALearner

    source = inspect.getsource(HyDRALearner.learn)

    assert "_prepare_callback" not in source
    assert ".start(" in source
    assert "self.callback" in source


def test_hydra_internal_helpers_do_not_accept_callback_parameters() -> None:
    from flowcean.hydra.learner import HyDRALearner

    learn_new_flow_params = inspect.signature(
        HyDRALearner.learn_new_flow,
    ).parameters
    discover_modes_params = inspect.signature(
        HyDRALearner.__dict__["_discover_modes"],
    ).parameters

    assert "callback" not in learn_new_flow_params
    assert "callback" not in discover_modes_params


def test_hydra_learner_populates_replay_on_success() -> None:
    from flowcean.hydra.learner import HyDRALearner
    from flowcean.hydra.model import HyDRAModel

    emitter = HyDRAReplayEmitter()
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
        callback=emitter,
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 0.1, 0.2, 0.3]}).lazy(),
        pl.DataFrame({"y": [0.0, 0.0, 0.0, 0.0]}).lazy(),
    )

    assert isinstance(model, HyDRAModel)
    assert emitter.replay is not None
    assert emitter.replay.final_mode_count == 1
    assert [step.kind for step in emitter.steps] == [
        "pending_segment_found",
        "candidate_window_evaluated",
        "candidate_window_evaluated",
        "accurate_segment_found",
        "mode_finalized",
    ]
    accurate_segment_step = emitter.steps[3]
    assert accurate_segment_step.mode_id == 0
    assert accurate_segment_step.threshold == 0.5


def test_hydra_learner_keeps_steps_without_final_replay_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flowcean.hydra.learner import HyDRALearner

    emitter = HyDRAReplayEmitter()
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
        callback=emitter,
    )

    def raise_selector_failure(*args: object, **kwargs: object) -> object:
        del args, kwargs
        message = "selector failed"
        raise RuntimeError(message)

    monkeypatch.setattr(learner, "_learn_selector", raise_selector_failure)

    with pytest.raises(RuntimeError, match="selector failed"):
        learner.learn(
            pl.DataFrame({"x": [0.0, 0.1, 0.2, 0.3]}).lazy(),
            pl.DataFrame({"y": [0.0, 0.0, 0.0, 0.0]}).lazy(),
        )

    assert emitter.replay is None
    assert [step.kind for step in emitter.steps] == [
        "pending_segment_found",
        "candidate_window_evaluated",
        "candidate_window_evaluated",
        "accurate_segment_found",
        "mode_finalized",
    ]


def test_hydra_learner_finalized_mode_uses_accepted_segment_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flowcean.hydra.learner import AccurateSegmentResult, HyDRALearner

    emitter = HyDRAReplayEmitter()
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
        callback=emitter,
    )

    call_count = 0

    def label_only_middle_segment(
        traces: list[pl.DataFrame],
        *args: object,
        **kwargs: object,
    ) -> AccurateSegmentResult:
        del args
        nonlocal call_count
        call_count += 1

        mode_id = kwargs["mode_id"]
        if call_count == 1:
            trace = traces[0].with_columns(
                pl.Series(
                    "mode",
                    [None, mode_id, mode_id, None],
                    dtype=pl.Int64,
                ),
            )
            accurate_rows = trace.slice(1, 2)
        else:
            trace = traces[0].with_columns(
                pl.Series(
                    "mode",
                    [
                        mode_id,
                        traces[0]["mode"][1],
                        traces[0]["mode"][2],
                        mode_id,
                    ],
                    dtype=pl.Int64,
                ),
            )
            accurate_rows = pl.concat([trace.slice(0, 1), trace.slice(3, 1)])

        return AccurateSegmentResult(
            updated_traces=[trace],
            accurate_rows=accurate_rows,
        )

    monkeypatch.setattr(
        "flowcean.hydra.learner.apply_model_to_traces",
        label_only_middle_segment,
    )

    learner.learn(
        pl.DataFrame({"x": [0.0, 0.1, 0.2, 0.3]}).lazy(),
        pl.DataFrame({"y": [0.0, 0.0, 0.0, 0.0]}).lazy(),
    )

    accurate_segment_step = emitter.steps[3]
    finalized_step = emitter.steps[4]

    assert accurate_segment_step.kind == "accurate_segment_found"
    assert accurate_segment_step.mode_id == 0
    assert accurate_segment_step.threshold == 0.5
    assert (
        accurate_segment_step.start_index,
        accurate_segment_step.end_index,
    ) == (1, 2)
    assert finalized_step.kind == "mode_finalized"
    assert finalized_step.mode_id == 0
    assert (finalized_step.start_index, finalized_step.end_index) == (1, 2)


def test_passive_circuit_example_uses_live_plot_callback_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    module = importlib.import_module("examples.passive_circuit.run")
    module_ast = ast.parse(inspect.getsource(module))
    main_ast = ast.parse(inspect.getsource(module.main))

    hydra_import_names = {
        alias.name
        for statement in ast.walk(module_ast)
        if isinstance(statement, ast.ImportFrom)
        and statement.module == "flowcean.hydra"
        for alias in statement.names
    }

    callback_names = {
        statement.targets[0].id
        for statement in ast.walk(main_ast)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "HyDRALivePlotCallback"
    }

    assert "HyDRALivePlotCallback" in hydra_import_names
    assert "HyDRAReplayEmitter" not in hydra_import_names
    assert callback_names
    assert any(
        isinstance(statement, ast.Call)
        and isinstance(statement.func, ast.Name)
        and statement.func.id == "HyDRALearner"
        and any(
            keyword.arg == "callback"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id in callback_names
            for keyword in statement.keywords
        )
        for statement in ast.walk(main_ast)
    )
    assert not any(
        isinstance(statement, ast.FunctionDef)
        and statement.name == "save_hydra_replay_plot"
        for statement in ast.walk(module_ast)
    )
    assert "plot_hydra_replay_step" not in hydra_import_names
    assert "HyDRAModel" not in hydra_import_names
