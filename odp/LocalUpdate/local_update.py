"""
Local Updating Procedure — Algorithm 1
----------------------------------------
Corrects an approximated N-D value function by re-running HJ Euler steps
only within detected leaking-corner regions and their causally connected
neighbourhood discovered by a BFS outward propagation.

Key design choices
------------------
* The Hamiltonian is supplied by the caller as ``hamiltonian_fn(V_slice, g)``.
  This keeps the algorithm fully dynamics-agnostic.
* Spatial derivatives inside the Hamiltonian are first-order upwind (see
  ``odp.LocalUpdate.spatial_derivs``).
* Time integration: CFL-stable explicit Euler with automatic sub-stepping.
  The stored tau step (e.g. 0.05 s) can violate the CFL stability condition
  (max_speed * dt / dx > 1), so each tau interval is split into n_substeps
  substeps of dt_sub = dt / n_substeps where CFL ≤ 1.
* BFS is vectorised using boolean NumPy masks (not Python sets) for
  efficiency.  At each BFS wave the entire ever-active region is re-integrated
  from the corrected source slice so that sub-stepping context is consistent.
* BRT semantics (minVWithV0): after each sub-step V_new is clamped against
  the initial target signed-distance — V_new = min(V_euler, target).

Reference: Algorithm_Local_Updating_Procedure.md, Equation (3)
"""

import math
import numpy as np
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_n_substeps(g, dt: float, max_speeds: Optional[List[float]] = None) -> int:
    """Return the minimum number of sub-steps that makes explicit Euler stable.

    The CFL condition for first-order upwind in dimension i is:

        max_speed_i * dt_sub / dx_i <= 1

    Rearranging: n_sub >= max_speed_i * dt / dx_i  for all i.

    Parameters
    ----------
    g          : Grid object (provides g.dx)
    dt         : outer tau step size
    max_speeds : optional per-dimension speed upper bounds.  If None the grid
                 spacing alone is used with a conservative default speed of 1.

    Returns
    -------
    int >= 1
    """
    if max_speeds is None:
        # Caller should provide this; fall back to dt/dx safely
        max_speeds = [1.0] * g.dims

    cfl_numbers = [
        abs(max_speeds[d]) * dt / float(g.dx[d])
        for d in range(g.dims)
    ]
    n_sub = max(1, math.ceil(max(cfl_numbers)))
    return n_sub


def _dilate_mask(mask: np.ndarray, periodic_dims: List[int]) -> np.ndarray:
    """Boolean morphological dilation by one cell in each dimension.

    For non-periodic dims the boundary wrap introduced by ``np.roll`` is
    suppressed so out-of-bounds neighbours are ignored.
    """
    result = mask.copy()
    ndim = mask.ndim
    for dim in range(ndim):
        fwd = np.roll(mask,  1, axis=dim)   # fwd[i] = mask[i-1]
        bwd = np.roll(mask, -1, axis=dim)   # bwd[i] = mask[i+1]
        if dim not in periodic_dims:
            # Suppress the wrap-around ghost at the boundaries
            lo = [slice(None)] * ndim;  lo[dim] = 0
            hi = [slice(None)] * ndim;  hi[dim] = -1
            fwd[tuple(lo)] = False
            bwd[tuple(hi)] = False
        result |= fwd | bwd
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def local_update(
    V_approx: np.ndarray,
    leaking_mask: np.ndarray,
    g,
    tau: np.ndarray,
    target: np.ndarray,
    hamiltonian_fn: Callable[[np.ndarray, object], np.ndarray],
    n_substeps: Optional[int] = None,
    max_speeds: Optional[List[float]] = None,
) -> np.ndarray:
    """Apply the Local Updating Procedure (Algorithm 1) to correct leaking corners.

    Parameters
    ----------
    V_approx : np.ndarray, shape (*spatial_shape, T)
        Full time-series approximated value function.

        **Time-axis convention** (matches ``HJSolver(..., saveAllTimeSteps=True)``):
        * ``V_approx[..., T-1]`` — initial condition / target set (t = 0).
        * ``V_approx[..., 0]``   — final BRS   (t = lookback_length).

    leaking_mask : np.ndarray of bool, shape (*spatial_shape, T)
        True at state/time pairs flagged as leaking corners.

    g : odp.Grid.Grid
        Spatial grid; provides ``g.dx``, ``g.pDim``.

    tau : np.ndarray, shape (T,)
        Monotone time array; ``dt = tau[1] - tau[0]``.

    target : np.ndarray, shape (*spatial_shape)
        Initial value function for minVWithV0 clamping (signed-distance target
        set, negative inside the target).

    hamiltonian_fn : callable ``(V_slice, g) -> H_array``
        Vectorised optimal Hamiltonian H*(x, ∇V) at every grid point.

    n_substeps : int, optional
        Number of CFL sub-steps per tau interval.  If None, auto-computed from
        ``max_speeds`` and grid spacings.

    max_speeds : list of float, optional
        Per-dimension upper bounds on the characteristic speed (|f_i|_max).
        Required for auto-computation of ``n_substeps``.  If both are None,
        a conservative value of ``ceil(max(dt/dx))`` is used.

    Returns
    -------
    V_corr : np.ndarray, shape (*spatial_shape, T)
        Corrected value function.  ``V_corr[..., T-1]`` equals
        ``V_approx[..., T-1]`` (initial condition unchanged).
    """
    spatial_shape = V_approx.shape[:-1]
    T             = V_approx.shape[-1]
    dt            = float(tau[1] - tau[0])
    periodic_dims: List[int] = list(g.pDim)

    # ── CFL sub-step size ───────────────────────────────────────────────────
    if n_substeps is None:
        n_substeps = _compute_n_substeps(g, dt, max_speeds)
    dt_sub = dt / n_substeps
    print(f"[local_update] n_substeps={n_substeps}, dt_sub={dt_sub:.5f} s")

    V_corr = V_approx.copy()

    for s_idx in range(T - 2, -1, -1):
        src_t = s_idx + 1                          # array index closer to IC
        V_src       = V_corr[..., src_t]           # corrected source slice
        V_approx_dst = V_approx[..., s_idx]        # approximate at destination

        leaking_s = leaking_mask[..., src_t]       # bool, *spatial_shape

        if not leaking_s.any():
            # No corrections needed; keep V_approx values
            continue

        # ── Sub-stepped Euler integration over the full spatial domain ───────
        # We integrate ALL grid cells (not just the leaking region) so that
        # spatial derivatives near the BFS frontier have correct neighbour
        # context.  The BFS below then selectively applies the corrected value
        # only to cells that genuinely need it.
        #
        # HJ PDE (backward time τ):  ∂V/∂τ = H*(x, ∇V)
        # Explicit Euler:            V(τ+δ) = V(τ) + dt_sub · H*
        #
        # Sign rationale: the HJSolver graph (graph_4D.py lines 113, 306)
        # computes V_new = f·∇V + diss_LF, then advances
        # V(τ+δ) = V(τ) + δ·V_new.  The BRS grows (V decreases at the
        # boundary) because H* < 0 there for uMode="max".
        V_sub = V_src.copy()
        for _ in range(n_substeps):
            H     = hamiltonian_fn(V_sub, g)
            V_sub = V_sub + dt_sub * H          # ← correct HJ sign
            V_sub = np.minimum(V_sub, target)   # minVWithV0

        # ── Vectorised BFS (wave-by-wave) ────────────────────────────────────
        # Start from the leaking-corner seed.  At each wave:
        #   1. Write the corrected value for every cell in the wave.
        #   2. Find which wave cells actually changed vs. the approximation.
        #   3. Dilate only from those changed cells → next frontier wave.
        # This prevents spurious expansion from previously-visited cells.
        V_dst   = V_approx_dst.copy()
        visited = leaking_s.copy()
        wave    = leaking_s.copy()

        while wave.any():
            # Apply corrected values to the current wave
            V_dst = np.where(wave, V_sub, V_dst)

            # Only cells in this wave that actually changed seed the next wave
            changed_in_wave = wave & (np.abs(V_sub - V_approx_dst) > 1e-10)
            dilated         = _dilate_mask(changed_in_wave, periodic_dims)
            next_wave       = dilated & ~visited

            visited |= next_wave
            wave     = next_wave

        V_corr[..., s_idx] = V_dst

        # Progress report every 5 outer steps
        step_num = T - 2 - s_idx
        if step_num % 5 == 0:
            pct   = 100.0 * step_num / max(T - 2, 1)
            n_vis = int(visited.sum())
            print(
                f"  [local_update] step {step_num:3d}/{T-2}"
                f"  ({pct:5.1f}%)"
                f"  visited={n_vis:7,}"
            )

    return V_corr
