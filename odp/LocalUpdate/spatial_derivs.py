"""
First-order upwind spatial derivatives for N-D value function arrays.

For each spatial dimension the module returns both the backward difference
(deriv_minus) and the forward difference (deriv_plus).  The caller is
responsible for selecting the upwind-appropriate side based on the sign of
the local characteristic speed (i.e. the dynamics f_i or optimal control).

Boundary treatment
------------------
* Periodic dimensions : wrap-around via ``np.roll`` (no special boundary fix).
* Non-periodic dims   : ghost cell = boundary value → one-sided difference
  at the boundary equals zero (equivalent to Neumann / no-flux condition).
  This is consistent with the convention used by the HeteroCL HJ solver,
  which also clamps spatial derivatives at grid boundaries.

Reference: Algorithm_Local_Updating_Procedure.md (Equation 3 — HJ_Update)
"""

import numpy as np
from typing import Tuple


def upwind_deriv(
    V: np.ndarray,
    g,
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """First-order backward and forward finite differences along ``dim``.

    Parameters
    ----------
    V   : np.ndarray
        Value function array whose shape matches ``g.pts_each_dim``.
    g   : odp.Grid.Grid
        Spatial grid object; provides ``g.dx``, ``g.pDim``.
    dim : int
        Dimension index (0-based) along which to differentiate.

    Returns
    -------
    deriv_minus : np.ndarray
        Backward difference: ``(V[i] - V[i-1]) / dx[dim]``.
    deriv_plus  : np.ndarray
        Forward difference:  ``(V[i+1] - V[i]) / dx[dim]``.

    Notes
    -----
    At non-periodic boundaries the ghost cell is replicated from the
    boundary value, making the one-sided derivative zero there.  This
    avoids artificial inflow of information from outside the grid.
    """
    dx = float(g.dx[dim])
    is_periodic = dim in g.pDim

    # Roll to get neighbour values
    V_prev = np.roll(V,  1, axis=dim)   # V[i-1]
    V_next = np.roll(V, -1, axis=dim)   # V[i+1]

    if is_periodic:
        deriv_minus = (V - V_prev) / dx
        deriv_plus  = (V_next - V) / dx
    else:
        # Fix left boundary: ghost at index -1 (rolled to position 0) = V[0]
        # Fix right boundary: ghost at index N (rolled to position -1) = V[-1]
        idx_left  = [slice(None)] * V.ndim
        idx_right = [slice(None)] * V.ndim
        idx_left[dim]  = slice(0, 1)
        idx_right[dim] = slice(-1, None)

        V_prev = V_prev.copy()
        V_next = V_next.copy()
        V_prev[tuple(idx_left)]  = V[tuple(idx_left)]   # deriv_minus[0] = 0
        V_next[tuple(idx_right)] = V[tuple(idx_right)]  # deriv_plus[-1] = 0

        deriv_minus = (V - V_prev) / dx
        deriv_plus  = (V_next - V) / dx

    return deriv_minus, deriv_plus
