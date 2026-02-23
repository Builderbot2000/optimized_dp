"""
Leaking Corner Detection
------------------------
A "leaking corner" in the decomposition context is a 4D state z where the
reconstructed value function V_approx(z) = max(V_1(z), V_2(z)) may deviate
from the true value V_true(z).  This happens when the subsystem-optimal
controls conflict and the max() reconstruction is underdetermined.

Detection heuristic (Threshold Strategy): states where the two subsystem
value slices are close — |V_1(z) - V_2(z)| < Δ — are flagged as leaking
because neither subsystem clearly dominates and small control perturbations
can flip the reconstruction.

Reference: Threshold_Strategy_Explanation.md
"""

import numpy as np


def detect_leaking_corners(
    V1_nd: np.ndarray,
    V2_nd: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Detect leaking corners in a max-reconstruction value function.

    Marks states where the two subsystem value estimates are within ``delta``
    of each other.  The result can be broadcast from per-time-step slices or
    applied to the full time-series in one call.

    Parameters
    ----------
    V1_nd : np.ndarray
        Value function from subsystem 1 (e.g. X-v-θ), single time step or
        full time-series.  Must be broadcastable against ``V2_nd``.
    V2_nd : np.ndarray
        Value function from subsystem 2 (e.g. Y-v-θ).  Same broadcasting
        rules apply.
    delta : float
        Threshold Δ.  States where |V1 - V2| < Δ are flagged as leaking.

    Returns
    -------
    np.ndarray of bool
        Same shape as ``np.maximum(V1_nd, V2_nd)`` after broadcasting.
        ``True``  → leaking corner (reconstruction uncertain).
        ``False`` → safe state (one subsystem clearly dominates).

    Examples
    --------
    Per-time-step (shapes (60,20,36) and (60,20,36)):

    >>> mask_t = detect_leaking_corners(V_x_t, V_y_t, delta=0.1)
    >>> mask_t.shape  # (60, 60, 20, 36) after broadcasting
    (60, 60, 20, 36)

    Full time-series (shapes (60,20,36,T) each):

    >>> mask = detect_leaking_corners(
    ...     result_x[:, None, :, :, :],
    ...     result_y[None, :, :, :, :],
    ...     delta=0.1,
    ... )
    >>> mask.shape  # (60, 60, 20, 36, T)
    (60, 60, 20, 36, 31)
    """
    diff = np.abs(V1_nd - V2_nd)
    return diff < delta
