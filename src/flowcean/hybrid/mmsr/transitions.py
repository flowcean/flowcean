"""Source-conditioned symbolic transition modeling from soft memberships."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import sympy as sp
from numpy.typing import NDArray

from ._search import (
    _aic,
    _Array,
    _Expression,
    _inputs,
    _search,
    _SearchFactory,
)
from .csr import MMSRCSR


@dataclass(frozen=True)
class MMSRTransition:
    """One ordered nonself edge with raw expression and zero-threshold guard.

    ``loss`` is the source-balanced weighted mean squared error of the sigmoid
    of ``expression`` against the soft successor destination membership.
    ``complexity`` counts the original raw search tree, not the fixed sigmoid.
    """

    source_location: int
    destination_location: int
    loss: float
    aic: float
    _function: _Expression = field(repr=False)

    @property
    def expression(self) -> sp.Expr:
        """Raw symbolic expression, without the training sigmoid wrapper."""
        return self._function.expression

    @property
    def complexity(self) -> int:
        """Original raw search-tree node count."""
        return self._function.complexity

    def guard(self, inputs: pl.DataFrame) -> NDArray[np.bool_]:
        """Evaluate ``expression >= 0`` at source-time regressors.

        Raises ``ValueError`` for nonfinite expression outputs. An active guard
        is not a physical reset or an event-localized continuous-time switch.
        """
        return (
            self._function.predict(
                _inputs(inputs, self._function.input_features)
            )
            >= 0
        )


@dataclass(frozen=True)
class MMSRTransitions:
    """All ordered nonself edges; no implicit self edges or execution policy.

    Multiple outgoing guards may be active together. Arbitration and
    probability propagation are deliberately not defined by this artifact.
    """

    input_features: tuple[str, ...]
    edges: tuple[MMSRTransition, ...]


def _pairs(
    traces: Sequence[pl.DataFrame], csr: MMSRCSR
) -> tuple[_Array, _Array, _Array]:
    if len(traces) != len(csr.memberships):
        raise ValueError("TM requires the original CSR traces in order")
    inputs, current, following = [], [], []
    for index, (trace, gamma) in enumerate(
        zip(traces, csr.memberships, strict=True)
    ):
        if gamma.shape != (trace.height, len(csr.expressions)):
            raise ValueError(
                f"TM trace {index}: CSR membership shape mismatch"
            )
        if (
            not np.isfinite(gamma).all()
            or np.any(gamma < 0)
            or np.any(gamma > 1)
            or not np.allclose(gamma.sum(axis=1), 1)
        ):
            raise ValueError(f"TM trace {index}: invalid CSR memberships")
        inputs.append(_inputs(trace, csr.input_features)[:-1])
        current.append(gamma[:-1])
        following.append(gamma[1:])
    if not inputs or sum(len(x) for x in inputs) == 0:
        raise ValueError("TM requires within-trace adjacent observations")
    return (
        np.concatenate(inputs),
        np.concatenate(current),
        np.concatenate(following),
    )


def _balanced_weights(current: _Array, following: _Array) -> _Array:
    departure = current * (1 - following)
    stay = current - departure
    departure_mass, stay_mass = float(departure.sum()), float(stay.sum())
    if departure_mass <= 1e-12 or stay_mass <= 1e-12:
        raise ValueError("Insufficient source departure/stay evidence")
    return departure + stay * (departure_mass / stay_mass)


def fit_mmsr_transitions(
    traces: Sequence[pl.DataFrame],
    csr: MMSRCSR,
    *,
    search_factory: _SearchFactory,
) -> MMSRTransitions:
    """Learn every source-conditioned next-location guard after CSR.

    Args:
        traces: The original ordered CSR traces. Adjacent rows are paired only
            within each independent trace; no timestamp or sampling period is
            inferred. Lengths and columns are checked, not dataset identity.
        csr: Independently fitted CSR artifact with soft row memberships.
        search_factory: Fresh configured ``pysr.PySRRegressor`` factory, with
            the same controlled settings as ``fit_mmsr_csr``. May use different
            budgets and operators from the CSR factory.

    Returns:
        All K(K-1) directed nonself guards. For each source k and destination j,
        the sigmoid of the raw expression at u_n targets gamma[j, n+1].
        Weights balance source departure against source stay, not destination
        classes; all outgoing edges of one source share the same weights.
        Selection uses ``2*c + N_pairs*log(weighted_mean_squared_error)``.

    Raises:
        ValueError: Missing departure/stay evidence or an entirely unusable
            edge frontier, with the source or edge identified. No partial edge
            collection is returned; the separate CSR artifact remains usable.

    Notes:
        This explicitly chooses the paper's source-time TM alignment: u_n and
        location n predict location n+1. The paper's formal mapping convention
        elsewhere differs. No reset map, continuous-time model conversion,
        initialization rule, or general multi-edge rollout is inferred.
    """
    inputs, current, following = _pairs(traces, csr)
    edges = []
    n_locations = len(csr.expressions)
    for source in range(n_locations):
        if n_locations == 1:
            break
        try:
            weights = _balanced_weights(
                current[:, source], following[:, source]
            )
        except ValueError as error:
            raise ValueError(
                f"MMSR transitions from location {source}: {error}"
            ) from error
        for destination in range(n_locations):
            if source == destination:
                continue
            frontier = _search(
                inputs,
                following[:, destination],
                weights,
                csr.input_features,
                search_factory,
                objective="transition",
            )
            if not frontier:
                raise ValueError(
                    f"MMSR transition {source}->{destination}: no usable candidate"
                )
            chosen = min(
                frontier,
                key=lambda c: _aic(c.function.complexity, len(inputs), c.loss),
            )
            edges.append(
                MMSRTransition(
                    source,
                    destination,
                    chosen.loss,
                    _aic(chosen.function.complexity, len(inputs), chosen.loss),
                    chosen.function,
                )
            )
    return MMSRTransitions(csr.input_features, tuple(edges))
