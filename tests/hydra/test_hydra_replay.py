from types import SimpleNamespace

import polars as pl
import pytest

from flowcean.hydra.replay import HyDRAReplayEmitter
from tests.hydra.test_hydra import AlwaysZeroLearner


class RecordingCallback:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _record(self, name: str, **payload: object) -> None:
        self.calls.append((name, payload))

    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        self._record(
            "start",
            trace_count=trace_count,
            threshold=threshold,
            start_width=start_width,
            step_width=step_width,
        )

    def pending_segment_found(
        self,
        trace_index: int,
        start_index: int,
        end_index: int,
    ) -> None:
        self._record(
            "pending_segment_found",
            trace_index=trace_index,
            start_index=start_index,
            end_index=end_index,
        )

    def candidate_window_evaluated(
        self,
        trace_index: int,
        start_index: int,
        end_index: int,
        window_size: int,
        fit: float,
    ) -> None:
        self._record(
            "candidate_window_evaluated",
            trace_index=trace_index,
            start_index=start_index,
            end_index=end_index,
            window_size=window_size,
            fit=fit,
        )

    def accurate_segment_found(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        mode_id: int,
        threshold: float,
    ) -> None:
        self._record(
            "accurate_segment_found",
            trace_index=trace_index,
            start_index=start_index,
            end_index=end_index,
            mode_id=mode_id,
            threshold=threshold,
        )

    def mode_finalized(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        mode_id: int,
    ) -> None:
        self._record(
            "mode_finalized",
            trace_index=trace_index,
            start_index=start_index,
            end_index=end_index,
            mode_id=mode_id,
        )

    def learning_stopped(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        reason: str,
    ) -> None:
        self._record(
            "learning_stopped",
            trace_index=trace_index,
            start_index=start_index,
            end_index=end_index,
            reason=reason,
        )

    def finish(self, *, final_mode_count: int) -> None:
        self._record("finish", final_mode_count=final_mode_count)


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


def test_hydra_learner_reports_callback_lifecycle_behavior() -> None:
    from flowcean.hydra.learner import HyDRALearner

    callback = RecordingCallback()
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=1,
        callback=callback,
    )

    learner.learn(
        pl.DataFrame({"x": [0.0, 0.1]}).lazy(),
        pl.DataFrame({"y": [0.0, 0.0]}).lazy(),
    )

    call_names = [name for name, _payload in callback.calls]
    assert call_names[0] == "start"
    assert "pending_segment_found" in call_names
    assert "candidate_window_evaluated" in call_names
    assert "accurate_segment_found" in call_names
    assert "mode_finalized" in call_names
    assert call_names[-1] == "finish"
    assert callback.calls[0][1] == {
        "trace_count": 1,
        "threshold": 0.5,
        "start_width": 2,
        "step_width": 1,
    }
    assert callback.calls[-1][1] == {"final_mode_count": 1}


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
