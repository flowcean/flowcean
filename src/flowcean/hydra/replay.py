from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

HyDRAStepKind = Literal[
    "pending_segment_found",
    "candidate_window_evaluated",
    "accurate_segment_found",
    "mode_finalized",
    "learning_stopped",
]


@dataclass(frozen=True)
class HyDRAStep:
    kind: HyDRAStepKind
    trace_index: int
    start_index: int
    end_index: int
    mode_id: int | None = None
    window_size: int | None = None
    fit: float | None = None
    threshold: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class HyDRAReplay:
    trace_count: int
    threshold: float
    start_width: int
    step_width: int
    final_mode_count: int
    steps: tuple[HyDRAStep, ...] = field(default_factory=tuple)


class HyDRAReplayEmitter:
    def __init__(self) -> None:
        self.replay: HyDRAReplay | None = None
        self.steps: list[HyDRAStep] = []
        self._trace_count: int | None = None
        self._threshold: float | None = None
        self._start_width: int | None = None
        self._step_width: int | None = None
        self._finished = False

    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        self.replay = None
        self.steps.clear()
        self._trace_count = trace_count
        self._threshold = threshold
        self._start_width = start_width
        self._step_width = step_width
        self._finished = False

    def _ensure_recording(self) -> None:
        if self._trace_count is None:
            message = "HyDRA replay emitter has not been started"
            raise RuntimeError(message)
        if self._finished:
            message = "HyDRA replay emitter is already finished"
            raise RuntimeError(message)

    def _metadata(self) -> tuple[int, float, int, int]:
        self._ensure_recording()
        trace_count = self._trace_count
        threshold = self._threshold
        start_width = self._start_width
        step_width = self._step_width
        if (
            trace_count is None
            or threshold is None
            or start_width is None
            or step_width is None
        ):
            message = "HyDRA replay emitter has incomplete metadata"
            raise RuntimeError(message)
        return (trace_count, threshold, start_width, step_width)

    def pending_segment_found(
        self,
        trace_index: int,
        start_index: int,
        end_index: int,
    ) -> None:
        self._ensure_recording()
        self.steps.append(
            HyDRAStep(
                "pending_segment_found",
                trace_index,
                start_index,
                end_index,
            ),
        )

    def candidate_window_evaluated(
        self,
        trace_index: int,
        start_index: int,
        end_index: int,
        window_size: int,
        fit: float,
    ) -> None:
        self._ensure_recording()
        self.steps.append(
            HyDRAStep(
                kind="candidate_window_evaluated",
                trace_index=trace_index,
                start_index=start_index,
                end_index=end_index,
                window_size=window_size,
                fit=fit,
            ),
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
        self._ensure_recording()
        self.steps.append(
            HyDRAStep(
                kind="accurate_segment_found",
                trace_index=trace_index,
                start_index=start_index,
                end_index=end_index,
                mode_id=mode_id,
                threshold=threshold,
            ),
        )

    def mode_finalized(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        mode_id: int,
    ) -> None:
        self._ensure_recording()
        self.steps.append(
            HyDRAStep(
                kind="mode_finalized",
                trace_index=trace_index,
                start_index=start_index,
                end_index=end_index,
                mode_id=mode_id,
            ),
        )

    def learning_stopped(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        reason: str,
    ) -> None:
        self._ensure_recording()
        self.steps.append(
            HyDRAStep(
                kind="learning_stopped",
                trace_index=trace_index,
                start_index=start_index,
                end_index=end_index,
                reason=reason,
            ),
        )

    def finish(self, *, final_mode_count: int) -> None:
        trace_count, threshold, start_width, step_width = self._metadata()
        self._finished = True
        self.replay = HyDRAReplay(
            trace_count=trace_count,
            threshold=threshold,
            start_width=start_width,
            step_width=step_width,
            final_mode_count=final_mode_count,
            steps=tuple(self.steps),
        )
