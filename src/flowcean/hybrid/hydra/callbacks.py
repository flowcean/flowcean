from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .learner import LearnedFlow, LearnedFlows, TraceSegment


class HyDRACallback:
    """Observe discovery geometry and scalar fits; override any desired hooks.

    Hooks run synchronously, and their exceptions propagate to the caller.
    """

    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        """Begin discovery over ``trace_count`` independent traces.

        ``threshold`` is the strict upper bound on absolute prediction error.
        ``start_width`` and ``step_width`` are the initial candidate window
        size and its growth step, measured in consecutive observations.
        """

    def pending_segment_found(self, segment: TraceSegment) -> None:
        """Report the unassigned segment where candidate fitting will begin."""

    def candidate_window_evaluated(
        self,
        *,
        segment: TraceSegment,
        fit: float,
    ) -> None:
        """Report candidate accuracy on the observations in ``segment``.

        ``fit`` is the maximum absolute prediction error over that window.
        """

    def candidate_selected(self, *, segment: TraceSegment, fit: float) -> None:
        """Report the last evaluated window whose candidate met the threshold.

        ``fit`` is its maximum absolute prediction error. The selected
        candidate will next be used to group matching unassigned observations.
        """

    def grouping_evaluated(
        self,
        *,
        flow_id: int,
        accepted_segments: tuple[TraceSegment, ...],
        considered_count: int,
    ) -> None:
        """Report candidate acceptance across all traces, before final refitting.

        Args:
            flow_id: ID reserved for this candidate if its final fit succeeds.
            accepted_segments: Previously unassigned observations that match
                the candidate within the error threshold.
            considered_count: Total number of unassigned observations across
                all traces before grouping, including both accepted and
                rejected observations. Already assigned observations are excluded.
        """

    def flow_finalized(self, *, flow_id: int, flow: LearnedFlow) -> None:
        """Report a flow successfully refitted on its accepted observations.

        ``flow_id`` is its position in ``result.flows``. ``flow`` contains the
        refitted model and the segments accepted before that refit.
        """

    def finish(self, result: LearnedFlows) -> None:
        """Receive the successful result assigning every supplied observation.

        ``result`` is the same object returned by ``learn``. This hook is not
        called if identification, a backend, or an earlier callback fails.
        """
