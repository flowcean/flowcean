from __future__ import annotations

from typing import Protocol


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
        trace_index: int,
        start_index: int,
        end_index: int,
    ) -> None: ...

    def candidate_window_evaluated(
        self,
        trace_index: int,
        start_index: int,
        end_index: int,
        window_size: int,
        fit: float,
    ) -> None: ...

    def accurate_segment_found(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        mode_id: int,
        threshold: float,
    ) -> None: ...

    def mode_finalized(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        mode_id: int,
    ) -> None: ...

    def learning_stopped(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        reason: str,
    ) -> None: ...

    def finish(self, *, final_mode_count: int) -> None: ...
