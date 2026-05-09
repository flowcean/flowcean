from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flowcean.hydra.learner import TraceSegment


@dataclass(frozen=True)
class HyDRACandidateFit:
    segment: TraceSegment
    window_size: int
    threshold: float
    fit: float
    actual_derivative: list[float]
    fitted_derivative: list[float]
    errors: list[float]


@dataclass(frozen=True)
class HyDRAGroupingTrace:
    trace_index: int
    row_indices: list[int]
    accepted_mask: list[bool]
    actual_derivative: list[float]
    fitted_derivative: list[float]
    errors: list[float]


@dataclass(frozen=True)
class HyDRAGroupingEvaluation:
    mode_id: int
    threshold: float
    triggering_segment: TraceSegment
    traces: tuple[HyDRAGroupingTrace, ...]


class HyDRACallback(Protocol):
    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None: ...

    def pending_segment_found(
        self,
        segment: TraceSegment,
    ) -> None: ...

    def candidate_window_evaluated(
        self,
        candidate: HyDRACandidateFit,
    ) -> None: ...

    def candidate_selected(self, candidate: HyDRACandidateFit) -> None: ...

    def grouping_evaluated(
        self,
        grouping: HyDRAGroupingEvaluation,
    ) -> None: ...

    def mode_finalized(
        self,
        *,
        mode_id: int,
        triggering_segment: TraceSegment,
        accepted_segments: Sequence[TraceSegment],
    ) -> None: ...

    def learning_stopped(
        self,
        *,
        segment: TraceSegment,
        reason: str,
    ) -> None: ...

    def finish(self, *, final_mode_count: int) -> None: ...


class LogCallback(HyDRACallback):
    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        print(
            f"HyDRA started: traces={trace_count}, threshold={threshold}, "
            f"start_width={start_width}, step_width={step_width}",
        )

    def pending_segment_found(self, segment: TraceSegment) -> None:
        print(
            "HyDRA pending segment: "
            f"trace={segment.trace_index}, "
            f"rows={segment.start_index}-{segment.end_index}",
        )

    def candidate_window_evaluated(
        self,
        candidate: HyDRACandidateFit,
    ) -> None:
        print(
            "HyDRA candidate window: "
            f"rows={candidate.segment.start_index}-{candidate.segment.end_index}, "
            f"fit={candidate.fit:.4f}, threshold={candidate.threshold:.4f}",
        )

    def candidate_selected(self, candidate: HyDRACandidateFit) -> None:
        print(
            "HyDRA candidate selected: "
            f"trace={candidate.segment.trace_index}, "
            f"rows={candidate.segment.start_index}-{candidate.segment.end_index}, "
            f"fit={candidate.fit:.4f}",
        )

    def grouping_evaluated(
        self,
        grouping: HyDRAGroupingEvaluation,
    ) -> None:
        accepted_count = sum(
            sum(trace.accepted_mask) for trace in grouping.traces
        )
        considered_count = sum(
            len(trace.row_indices) for trace in grouping.traces
        )
        print(
            f"HyDRA grouping: mode={grouping.mode_id}, "
            f"accepted_rows={accepted_count}/{considered_count}",
        )

    @override
    def mode_finalized(
        self,
        *,
        mode_id: int,
        triggering_segment: TraceSegment,
        accepted_segments: Sequence[TraceSegment],
    ) -> None:
        _ = triggering_segment
        print(
            f"HyDRA mode finalized: mode={mode_id}, "
            f"segments={len(accepted_segments)}",
        )

    def learning_stopped(
        self,
        *,
        segment: TraceSegment,
        reason: str,
    ) -> None:
        print(
            "HyDRA stopped: "
            f"trace={segment.trace_index}, "
            f"rows={segment.start_index}-{segment.end_index}, "
            f"reason={reason}",
        )

    def finish(self, *, final_mode_count: int) -> None:
        print(f"HyDRA finished: modes={final_mode_count}")


class NoOpCallback(HyDRACallback):
    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        pass

    def pending_segment_found(
        self,
        segment: TraceSegment,
    ) -> None:
        pass

    def candidate_window_evaluated(
        self,
        candidate: HyDRACandidateFit,
    ) -> None:
        pass

    def candidate_selected(self, candidate: HyDRACandidateFit) -> None: ...

    def grouping_evaluated(
        self,
        grouping: HyDRAGroupingEvaluation,
    ) -> None:
        pass

    def mode_finalized(
        self,
        *,
        mode_id: int,
        triggering_segment: TraceSegment,
        accepted_segments: Sequence[TraceSegment],
    ) -> None:
        pass

    def learning_stopped(
        self,
        *,
        segment: TraceSegment,
        reason: str,
    ) -> None:
        pass

    def finish(self, *, final_mode_count: int) -> None:
        pass
