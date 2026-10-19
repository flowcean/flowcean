"""Exact scalar DTW comparison and ordered FaMoS clustering.

The backend uses absolute local costs and its own optimal-path tie breaking;
bitwise equivalence with MATLAB's DTW paths is not promised. Algorithm
adapted from FaMoS-DT; see LICENSE-FaMoS-DT for its MIT license.
"""

import numpy as np
from dtaidistance import dtw

from ._data import FloatArray


def _alignment_score(a: FloatArray, b: FloatArray) -> float:
    result = dtw.warping_paths_fast(
        np.ascontiguousarray(a, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
        inner_dist="euclidean",
        use_pruning=False,
    )
    cost, matrix = result[0], result[1]
    path = dtw.best_path(matrix)
    indices = np.asarray(path, dtype=np.float64)
    if np.any(np.std(indices, axis=0) == 0):
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(indices[:, 0], indices[:, 1])[0, 1])
    return 0.5 * float(cost) / len(path) + 0.5 * (1 - correlation)


def segment_score(a: FloatArray, b: FloatArray, *, raw: bool) -> float:
    """Compare start/end-matched shorter windows, preserving derivative offsets."""
    length = min(len(a), len(b))
    start_a, start_b = a[:length], b[:length]
    end_a, end_b = a[-length:], b[-length:]
    if raw:
        start_a, start_b = start_a - start_a[0], start_b - start_b[0]
        end_a, end_b = end_a - end_a[-1], end_b - end_b[-1]
    return min(
        _alignment_score(start_a, start_b),
        _alignment_score(end_a, end_b),
    )


def ordered_clusters(
    scores: FloatArray, *, alpha: float, rho_min: float, rho_max: float
) -> list[int]:
    """Asymmetric row thresholds; reassign a segment, never merge clusters."""
    count = len(scores)
    labels = [-1] * count
    confidence = np.full(count, np.inf)
    for i in range(count):
        if labels[i] == -1:
            labels[i] = i
        if count == 1:
            continue
        nearest = np.min(np.delete(scores[i], i))
        threshold = np.clip(alpha * nearest, rho_min, rho_max)
        # An assigned self-pair changes neither membership nor confidence.
        for j in range(i + 1, count):
            if scores[i, j] <= threshold and (
                labels[j] == -1 or scores[i, j] < confidence[j]
            ):
                labels[j] = labels[i]
                confidence[j] = scores[i, j]
    compact = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    return [compact[label] for label in labels]
