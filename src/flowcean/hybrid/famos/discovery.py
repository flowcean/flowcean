"""Geometric change detection and per-signal DTW location discovery.

Algorithm adapted from FaMoS-DT (TUHH Institute of Embedded Systems);
see the bundled LICENSE-FaMoS-DT for the original MIT license.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import polars as pl
from scipy.signal import find_peaks

from ._data import FloatArray, columns, sampling_period
from ._dtw import ordered_clusters, segment_score


@dataclass(frozen=True)
class FamosSegment:
    """One original-frame half-open interval, including unassigned evidence.

    Attributes:
        trace_index: Index in the independent training-trace sequence.
        start: First original row, inclusive.
        stop: Last original row, exclusive.
        location_id: Global location ID, or None for derivative warmup or
            a segment too short for trimmed DTW comparison.
    """

    trace_index: int
    start: int
    stop: int
    location_id: int | None


@dataclass(frozen=True)
class FamosDiscovery:
    """Discovery independent of dynamics or transition fitting.

    Attributes:
        segments: Intervals in trace/row order. Boundaries remain distinct
            even when adjacent intervals have the same location.
        trace_lengths: Original training-frame lengths, including warmup.
        input_features: Input column names in vector order.
        output_features: Output column names in vector order.
        time_column: Timestamp column name.
        dt: Shared sampling period.
        input_scales: Training maximum absolute inputs, with zero replaced
            by one. Prediction uses these scales without clipping.
        output_scales: Corresponding training maximum absolute outputs.
            This follows the artifact's evaluation entry point; its trace
            entry point instead uses max-positive output scaling.
        location_keys: Global IDs index tuples of per-signal cluster IDs,
            ordered as outputs followed by inputs.
    """

    segments: tuple[FamosSegment, ...]
    trace_lengths: tuple[int, ...]
    input_features: tuple[str, ...]
    output_features: tuple[str, ...]
    time_column: str
    dt: float
    input_scales: FloatArray
    output_scales: FloatArray
    location_keys: tuple[tuple[int, ...], ...]

    def to_labeled_frames(
        self, traces: Sequence[pl.DataFrame]
    ) -> list[pl.DataFrame]:
        """Append nullable location_id to the original training frames.

        Supply the same observations, in the same trace and row order used
        for discovery. This is a provenance projection, not a classifier
        for new traces. Frames are not resampled or truncated. An existing
        location_id column raises ValueError rather than being overwritten.
        """
        if any("location_id" in trace.columns for trace in traces):
            raise ValueError(
                "Cannot append location_id: column already exists"
            )
        self._check_traces(traces)
        labels: list[list[int | None]] = [
            [None] * length for length in self.trace_lengths
        ]
        for segment in self.segments:
            labels[segment.trace_index][segment.start : segment.stop] = [
                segment.location_id
            ] * (segment.stop - segment.start)
        return [
            trace.with_columns(pl.Series("location_id", ids, dtype=pl.Int64))
            for trace, ids in zip(traces, labels, strict=True)
        ]

    def _check_traces(self, traces: Sequence[pl.DataFrame]) -> None:
        if tuple(trace.height for trace in traces) != self.trace_lengths:
            raise ValueError("Supply the original discovery traces in order")
        for trace in traces:
            sampling_period(trace, self.time_column, self.dt)
            columns(trace, (*self.output_features, *self.input_features))


def _merge_peaks(peaks: Sequence[int], distance: float) -> list[int]:
    retained: list[int] = []
    for peak in sorted(set(peaks)):
        if not retained or peak - retained[-1] > distance:
            retained.append(peak)
    return retained


def _geometric_peaks(
    derivatives: Sequence[FloatArray], window: int, peak_height: float
) -> list[int]:
    def scan(start: int, stop: int, level: int) -> list[int]:
        if stop - start < 2 * window + 2:
            return []
        signal = derivatives[level]
        candidates = np.arange(start + window, stop - window)
        distances = np.array(
            [
                np.sum(
                    np.abs(
                        (signal[t - window : t] - signal[t - window])
                        - (signal[t + 1 : t + window + 1] - signal[t + 1])
                    )
                )
                for t in candidates
            ]
        )
        # Match the artifact's leading zero padding, without trailing padding.
        detected = find_peaks(
            np.concatenate((np.zeros(window), distances)), height=peak_height
        )[0]
        peaks = _merge_peaks((detected + start).tolist(), 1.5 * window)
        if level + 1 < len(derivatives):
            inset = window // 2
            starts = [start, *(peak + inset for peak in peaks)]
            stops = [*(peak - inset + 1 for peak in peaks), stop]
            children = []
            for left, right in zip(starts, stops, strict=True):
                children.extend(scan(left, right, level + 1))
            peaks.extend(children)
        return sorted(set(peaks))

    return scan(0, len(derivatives[0]), 0)


def discover_famos_locations(
    traces: Sequence[pl.DataFrame],
    *,
    input_features: Sequence[str],
    output_features: Sequence[str],
    window: int,
    trim: int,
    peak_height: float,
    rho_min: float,
    rho_max: float,
    time_column: str = "t",
    max_derivative: int = 1,
    clustering_derivative: int = 1,
    alpha: float = 2.0,
) -> FamosDiscovery:
    """Discover sampled locations without fitting any dynamics.

    Each signal is segmented geometrically, then clustered by exact DTW.
    Global locations are tuples of per-signal clusters, not connected
    components of a distance graph. Ordered asymmetric reassignment follows
    trace/row order. The DTW backend is dtaidistance's exact C implementation;
    MATLAB optimal-path tie equivalence is not claimed.

    Args:
        traces: Independent, uniformly sampled eager frames. Only selected
            columns are read; differences and histories restart per trace.
        input_features: Inputs used raw for detection and clustering.
        output_features: Outputs used raw and through max_derivative for
            recursive detection, and at clustering_derivative for DTW.
        window: Rows on EACH side of a candidate, excluding its center.
            Each window is offset by its first value before L1 comparison.
            Recursion insets peak-adjacent ends by floor(window / 2),
            retaining the outer trace endpoints.
        trim: Rows removed at each segment end for DTW only. Fewer than
            two remaining rows leaves the segment unassigned.
        peak_height: Minimum geometric L1 peak height in scaled signals.
        rho_min: Lower bound on a row's DTW acceptance threshold.
        rho_max: Upper bound on that threshold.
        time_column: Timestamp column, not a regression feature.
        max_derivative: Highest repeated backward difference divided by dt.
            The first max_derivative rows remain unassigned, not padded.
        clustering_derivative: Output derivative used for DTW; zero removes
            offsets from start/end-matched windows, higher orders retain them.
        alpha: Threshold multiplier of each segment's nearest-other score,
            clipped to [rho_min, rho_max].

    Returns:
        Partial location labels and training-only scales, retaining original
        trace boundaries. No accuracy, rank, or full-coverage gate is applied.
    """
    inputs, outputs = tuple(input_features), tuple(output_features)
    names = (*outputs, *inputs, time_column)
    if not outputs or len(names) != len(set(names)) or "location_id" in names:
        raise ValueError("Select distinct input/output/time names and outputs")
    if not traces:
        raise ValueError("Supply at least one training trace")
    if (
        window < 2
        or trim < 0
        or not 0 <= clustering_derivative <= max_derivative
    ):
        raise ValueError("Invalid window, trim, or derivative orders")
    if not 0 <= rho_min <= rho_max or alpha <= 0 or peak_height < 0:
        raise ValueError("Invalid peak or clustering thresholds")
    dt = sampling_period(traces[0], time_column)
    observations = []
    for trace in traces:
        sampling_period(trace, time_column, dt)
        observations.append(columns(trace, (*outputs, *inputs)))
    scales = np.max(np.abs(np.concatenate(observations)), axis=0)
    scales[scales == 0] = 1
    # Per signal: (trace index, start, stop, trimmed comparison signal).
    local_segments: list[list[tuple[int, int, int, FloatArray]]] = [
        [] for _ in scales
    ]
    global_boundaries: list[list[int]] = []
    for trace_index, observation in enumerate(observations):
        length = len(observation)
        warmup = min(max_derivative, length)
        scaled = observation / scales
        signals = []
        local_peaks = []
        for signal_index in range(len(scales)):
            derivative = scaled[:, signal_index]
            derivatives = [derivative[warmup:]]
            if signal_index < len(outputs):
                for level in range(1, max_derivative + 1):
                    derivative = np.diff(derivative) / dt
                    derivatives.append(derivative[max_derivative - level :])
            signals.append(derivatives)
            local_peaks.append(
                _geometric_peaks(derivatives, window, peak_height)
            )
        merged = _merge_peaks(
            [p for peaks in local_peaks for p in peaks], window
        )
        global_boundaries.append(
            sorted({0, warmup, length, *(p + warmup for p in merged)})
        )
        for signal_index, peaks in enumerate(local_peaks):
            snapped = {
                min(merged, key=lambda p: (abs(p - peak), p)) for peak in peaks
            }
            boundaries = sorted(
                {warmup, length, *(p + warmup for p in snapped)}
            )
            level = clustering_derivative if signal_index < len(outputs) else 0
            signal = signals[signal_index][level]
            for start, stop in pairwise(boundaries):
                comparison = signal[
                    start - warmup + trim : max(
                        start - warmup + trim, stop - warmup - trim
                    )
                ]
                local_segments[signal_index].append(
                    (trace_index, start, stop, comparison)
                )
    memberships = [
        np.full((len(obs), len(scales)), -1, dtype=int) for obs in observations
    ]
    for signal_index, segments in enumerate(local_segments):
        usable = [segment for segment in segments if len(segment[3]) >= 2]
        scores = np.zeros((len(usable), len(usable)))
        raw = signal_index >= len(outputs) or clustering_derivative == 0
        for i, first in enumerate(usable):
            for j in range(i + 1, len(usable)):
                scores[i, j] = scores[j, i] = segment_score(
                    first[3], usable[j][3], raw=raw
                )
        labels = ordered_clusters(
            scores, alpha=alpha, rho_min=rho_min, rho_max=rho_max
        )
        for (trace_index, start, stop, _), label in zip(
            usable, labels, strict=True
        ):
            memberships[trace_index][start:stop, signal_index] = label
    keys: dict[tuple[int, ...], int] = {}
    result = []
    for trace_index, boundaries in enumerate(global_boundaries):
        for start, stop in pairwise(boundaries):
            key = tuple(
                int(label) for label in memberships[trace_index][start]
            )
            location = None if -1 in key else keys.setdefault(key, len(keys))
            result.append(FamosSegment(trace_index, start, stop, location))
    return FamosDiscovery(
        segments=tuple(result),
        trace_lengths=tuple(len(obs) for obs in observations),
        input_features=inputs,
        output_features=outputs,
        time_column=time_column,
        dt=dt,
        input_scales=scales[len(outputs) :],
        output_scales=scales[: len(outputs)],
        location_keys=tuple(keys),
    )
