"""Multi-modal symbolic regression for known-K scalar input-output mappings.

CSR jointly learns symbolic expressions, soft memberships, and variances.
Subsequent transition modeling learns source-conditioned symbolic guards.
Both stages return inspectable artifacts, not a universal hybrid runtime.
PySR/Julia are loaded only when fitting, via the ``pysr`` optional extra.
"""

from .csr import MMSRCSR, CSRDiagnostics, fit_mmsr_csr
from .transitions import MMSRTransition, MMSRTransitions, fit_mmsr_transitions

__all__ = (
    "MMSRCSR",
    "CSRDiagnostics",
    "MMSRTransition",
    "MMSRTransitions",
    "fit_mmsr_csr",
    "fit_mmsr_transitions",
)
