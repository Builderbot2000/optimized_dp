import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import CylinderShape
from odp.Plots import PlotOptions, visualize_plots
from odp.solver import HJSolver
from odp.dynamics import DubinsCar4D2, DubinsCarXSubsystem, DubinsCarYSubsystem
from odp.LocalUpdate import detect_leaking_corners, upwind_deriv, local_update

# ---------------------------------------------------------------------------
# Shared solver settings
# ---------------------------------------------------------------------------

lookback_length = 1.5
t_step = 0.05
small_number = 1e-5

tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# Backward Reachable Tube computation method
compMethods = {"TargetSetMode": "minVWithV0"}

# ---------------------------------------------------------------------------
# Section 1: Direct 4D solve (baseline)
#
# State order: (x, y, v, theta)
# Grid bounds:  x in [-3, 3], y in [-1, 4], v in [0, 4], theta in [-pi, pi]
# Periodic dim: theta (dim index 3, 0-based)
# Target set:   2D cylinder in (x, y), ignoring v and theta,
#               centred at (x=0, y=2) with radius 0.8
# ---------------------------------------------------------------------------

g_4d = Grid(
    np.array([-3.0, -1.0, 0.0, -math.pi]),
    np.array([3.0,   4.0, 4.0,  math.pi]),
    4,
    np.array([60, 60, 20, 36]),
    [3],  # theta is periodic
)

car_4d = DubinsCar4D2(uMode="max", dMode="min")

target_4d = CylinderShape(g_4d, [2, 3], np.array([0., 2., 0., 0.]), 0.8)

result_4d = HJSolver(
    car_4d, g_4d, target_4d, tau, compMethods,
    saveAllTimeSteps=True, accuracy="medium",
)

# BRS at the final look-back time; shape: (60, 60, 20, 36)
last_time_step_4d = result_4d[..., 0]

# ---------------------------------------------------------------------------
# Section 2: X–velocity–heading subsystem solve
#
# State order: (x, v, theta)
# Grid bounds:  x in [-3, 3], v in [0, 4], theta in [-pi, pi]
# Periodic dim: theta (dim index 2, 0-based)
# Target set:   1D slab in x, ignoring v and theta,
#               centred at x=0 with radius 0.8
# Disturbance:  d_x in [-0.25, 0.25]
# ---------------------------------------------------------------------------

g_x = Grid(
    np.array([-3.0, 0.0, -math.pi]),
    np.array([3.0,  4.0,  math.pi]),
    3,
    np.array([60, 20, 36]),
    [2],  # theta is periodic
)

car_x = DubinsCarXSubsystem(
    uMin=[-1.5, -math.pi / 18],
    uMax=[1.5,   math.pi / 18],
    dMin=[-0.25],
    dMax=[0.25],
    uMode="max",
    dMode="min",
)

target_x = CylinderShape(g_x, [1, 2], np.array([0., 0., 0.]), 0.8)

result_x = HJSolver(
    car_x, g_x, target_x, tau, compMethods,
    saveAllTimeSteps=True, accuracy="medium",
)

# BRS at the final look-back time; shape: (60, 20, 36)
V_x = result_x[..., 0]

# ---------------------------------------------------------------------------
# Section 3: Y–velocity–heading subsystem solve
#
# State order: (y, v, theta)
# Grid bounds:  y in [-1, 4], v in [0, 4], theta in [-pi, pi]
# Periodic dim: theta (dim index 2, 0-based)
# Target set:   1D slab in y, ignoring v and theta,
#               centred at y=2 with radius 0.8
# Disturbance:  d_y in [-0.25, 0.25]
# ---------------------------------------------------------------------------

g_y = Grid(
    np.array([-1.0, 0.0, -math.pi]),
    np.array([4.0,  4.0,  math.pi]),
    3,
    np.array([60, 20, 36]),
    [2],  # theta is periodic
)

car_y = DubinsCarYSubsystem(
    uMin=[-1.5, -math.pi / 18],
    uMax=[1.5,   math.pi / 18],
    dMin=[-0.25],
    dMax=[0.25],
    uMode="max",
    dMode="min",
)

target_y = CylinderShape(g_y, [1, 2], np.array([2., 0., 0.]), 0.8)

result_y = HJSolver(
    car_y, g_y, target_y, tau, compMethods,
    saveAllTimeSteps=True, accuracy="medium",
)

# BRS at the final look-back time; shape: (60, 20, 36)
V_y = result_y[..., 0]

# ---------------------------------------------------------------------------
# Section 4: Reconstruct 4D BRS from the two 3D subsystem BRS values
#
# Per the decomposition design (Section 6):
#   V_4d(x_i, y_j, v_k, theta_l) = max( V_x(x_i, v_k, theta_l),
#                                        V_y(y_j, v_k, theta_l) )
#
# Broadcasting layout:
#   V_x reshaped to (60,  1, 20, 36)  [x,  1, v, theta]
#   V_y reshaped to ( 1, 60, 20, 36)  [1,  y, v, theta]
#   V_4d shape:    (60, 60, 20, 36)  [x,  y, v, theta]
# ---------------------------------------------------------------------------

V_4d = np.maximum(
    V_x[:, None, :, :],   # (60,  1, 20, 36)
    V_y[None, :, :, :],   # ( 1, 60, 20, 36)
)

# Sanity check: both results should have the same spatial shape as the 4D grid
assert last_time_step_4d.shape == V_4d.shape == (60, 60, 20, 36), (
    f"Shape mismatch: last_time_step_4d={last_time_step_4d.shape}, "
    f"V_4d={V_4d.shape}"
)

print("4D direct BRS shape:         ", last_time_step_4d.shape)
print("Reconstructed 4D BRS shape:  ", V_4d.shape)
print("Done. Results held in memory as `last_time_step_4d` and `V_4d`.")

# ---------------------------------------------------------------------------
# Section 5: Visualization
#
# Plot the BRS as a zero-sublevel-set isosurface in the (x, y, theta) slice
# at v-index 10, matching the original dubins_4d_avoid.py example.
#
# result_4d has a time axis  -> animated isosurface (BRT growing over time)
# V_4d has no time axis     -> static isosurface  (final BRS only)
# ---------------------------------------------------------------------------

# Direct 4D BRS (animated over look-back time)
po_4d = PlotOptions(
    do_plot=True,
    plot_type="set",
    plotDims=[0, 1, 3],
    slicesCut=[10],
    save_fig=True,
    filename="brs_4d_direct.png",
)
visualize_plots(result_4d, g_4d, po_4d)

# Reconstructed 4D BRS from the two 3D subsystem solves (static, final time)
po_decomp = PlotOptions(
    do_plot=True,
    plot_type="set",
    plotDims=[0, 1, 3],
    slicesCut=[10],
    save_fig=True,
    filename="brs_4d_decomposed.png",
)
visualize_plots(V_4d, g_4d, po_decomp)

# ---------------------------------------------------------------------------
# Section 6: Full time-series 4D approximation
#
# result_x and result_y both have shape (*spatial, T) where T = len(tau) = 31.
# Build V_approx of shape (60, 60, 20, 36, 31) by broadcasting:
#   result_x[:, None, :, :, :]  →  (60,  1, 20, 36, 31)
#   result_y[None, :, :, :, :]  →  ( 1, 60, 20, 36, 31)
# ---------------------------------------------------------------------------

V_approx = np.maximum(
    result_x[:, None, :, :, :],   # (60,  1, 20, 36, 31)
    result_y[None, :, :, :, :],   # ( 1, 60, 20, 36, 31)
)                                  # → (60, 60, 20, 36, 31)

assert V_approx.shape == result_4d.shape == (60, 60, 20, 36, len(tau)), (
    f"Time-series shape mismatch: V_approx={V_approx.shape}, "
    f"result_4d={result_4d.shape}"
)
print("Full time-series V_approx shape:", V_approx.shape)

# ---------------------------------------------------------------------------
# Section 7: Detect leaking corners
#
# Threshold Δ (user-facing parameter):
#   Increase Δ to flag more states as leaking (more conservative correction).
#   Decrease Δ to flag fewer states (faster, may miss some errors).
#
# leaking_mask shape: (60, 60, 20, 36, 31) — True where |V_x - V_y| < DELTA
# ---------------------------------------------------------------------------

DELTA_THRESH = 0.1

leaking_mask = detect_leaking_corners(
    result_x[:, None, :, :, :],   # (60,  1, 20, 36, 31)
    result_y[None, :, :, :, :],   # ( 1, 60, 20, 36, 31)
    delta=DELTA_THRESH,
)                                  # → (60, 60, 20, 36, 31) bool

n_total   = leaking_mask.size
n_leaking = int(leaking_mask.sum())
print(
    f"Leaking corners: {n_leaking:,} / {n_total:,}  "
    f"({100.0 * n_leaking / n_total:.1f}%),  Δ = {DELTA_THRESH}"
)

# ---------------------------------------------------------------------------
# Section 8: Dynamics-specific Hamiltonian for DubinsCar4D2
#
# H*(x, ∇V) = max_u  min_d  ∇V · f(x, u, d)
#
# DubinsCar4D2 dynamics (uMode="max", dMode="min", dMin=dMax=[0,0]):
#
#   x_dot     = v·cos(θ)          (no disturbance in the 4D experiment)
#   y_dot     = v·sin(θ)
#   v_dot     = a*                 a* = uMax[0] if ∂V/∂v ≥ 0 else uMin[0]
#   θ_dot     = v·tan(δ*)/L       δ* = uMax[1] if ∂V/∂θ ≥ 0 else uMin[1]
#
# Upwind rule per dimension:
#   x : use backward diff if v·cos(θ) ≥ 0, forward diff otherwise
#   y : use backward diff if v·sin(θ) ≥ 0, forward diff otherwise
#   v : determine a* from central pv estimate; then re-select upwind based on
#       sign of a* (backward if a* > 0, forward if a* < 0)
#   θ : determine θ_dot* = v·tan(δ*)/L from central pθ; upwind based on sign
# ---------------------------------------------------------------------------

_L = 0.3  # wheelbase constant (Tamiya TT02, matches DubinsCar4D2)


def dubins4d2_hamiltonian(V_slice: np.ndarray, g) -> np.ndarray:
    """Vectorised optimal Hamiltonian for the 4D Dubins car (no disturbance).

    Parameters
    ----------
    V_slice : np.ndarray, shape (60, 60, 20, 36)
        Value function at a single time step on the 4D grid.
    g : odp.Grid.Grid
        The 4D grid (g_4d).

    Returns
    -------
    H : np.ndarray, shape (60, 60, 20, 36)
        H*(x, ∇V) at every grid point.
    """
    uMin = car_4d.uMin   # [-1.5, -π/18]
    uMax = car_4d.uMax   # [ 1.5,  π/18]

    v_grid     = g.vs[2]   # shape (1, 1, 20,  1)
    theta_grid = g.vs[3]   # shape (1, 1,  1, 36)

    cos_theta = np.cos(theta_grid)
    sin_theta = np.sin(theta_grid)

    # Spatial derivatives (both sides) for all four dimensions
    dm0, dp0 = upwind_deriv(V_slice, g, 0)   # ∂V/∂x
    dm1, dp1 = upwind_deriv(V_slice, g, 1)   # ∂V/∂y
    dm2, dp2 = upwind_deriv(V_slice, g, 2)   # ∂V/∂v
    dm3, dp3 = upwind_deriv(V_slice, g, 3)   # ∂V/∂θ

    # --- x dimension ---
    # flow_x = v·cos(θ);  no disturbance (dMin=dMax=0)
    flow_x = v_grid * cos_theta
    px = np.where(flow_x >= 0, dm0, dp0)

    # --- y dimension ---
    flow_y = v_grid * sin_theta
    py = np.where(flow_y >= 0, dm1, dp1)

    H = flow_x * px + flow_y * py

    # --- v dimension ---
    # Max-player picks a* = uMax if pv ≥ 0, uMin if pv < 0.
    # Use central diff to determine sign of pv, then re-select upwind side.
    pv_central = 0.5 * (dm2 + dp2)
    a_opt = np.where(pv_central >= 0, uMax[0], uMin[0])
    pv = np.where(a_opt > 0, dm2, np.where(a_opt < 0, dp2, pv_central))
    H += a_opt * pv

    # --- θ dimension ---
    # Max-player picks δ* = uMax[1] if pθ ≥ 0, uMin[1] if pθ < 0.
    # θ_dot = v·tan(δ*)/L; upwind based on sign of θ_dot.
    ptheta_central = 0.5 * (dm3 + dp3)
    delta_opt = np.where(ptheta_central >= 0, uMax[1], uMin[1])
    theta_dot = v_grid * np.tan(delta_opt) / _L
    ptheta = np.where(theta_dot > 0, dm3, np.where(theta_dot < 0, dp3, ptheta_central))
    H += theta_dot * ptheta

    return H


# ---------------------------------------------------------------------------
# Section 9: Local Updating Procedure (Algorithm 1)
#
# Corrects V_approx in-place within detected leaking-corner islands and their
# causally connected BFS neighbourhood.  Only cells whose corrected value
# differs from the approximation propagate the update outward.
# ---------------------------------------------------------------------------

print("\nRunning Local Updating Procedure …")

# Max characteristic speeds per dimension (used to ensure CFL stability):
#   dim 0 (x)    : v_max * |cos(θ)| ≤ v_max
#   dim 1 (y)    : v_max * |sin(θ)| ≤ v_max
#   dim 2 (v)    : |a|_max = uMax[0]
#   dim 3 (theta): v_max * |tan(delta_max)| / L
_v_max = float(g_4d.max[2])                                            # 4.0
_theta_dot_max = _v_max * math.tan(abs(car_4d.uMax[1])) / _L          # ≈ 2.35
_max_speeds = [_v_max, _v_max, car_4d.uMax[0], _theta_dot_max]

V_corr = local_update(
    V_approx     = V_approx,
    leaking_mask = leaking_mask,
    g            = g_4d,
    tau          = tau,
    target       = target_4d,
    hamiltonian_fn = dubins4d2_hamiltonian,
    max_speeds   = _max_speeds,
)
print("Local Updating Procedure complete.\n")

assert V_corr.shape == (60, 60, 20, 36, len(tau)), (
    f"Unexpected V_corr shape: {V_corr.shape}"
)

# ---------------------------------------------------------------------------
# Section 10: Verification and reporting
#
# Compare L-inf errors before and after the local correction against the
# direct 4D HJ solve (last_time_step_4d).
# ---------------------------------------------------------------------------

V_approx_final   = V_approx[..., 0]     # shape (60, 60, 20, 36)
V_corrected      = V_corr[..., 0]       # shape (60, 60, 20, 36)

err_before = float(np.max(np.abs(V_approx_final - last_time_step_4d)))
err_after  = float(np.max(np.abs(V_corrected   - last_time_step_4d)))

print("=" * 60)
print("Error vs direct 4D solve (L-inf norm on final BRS slice)")
print(f"  Before correction : {err_before:.6f}")
print(f"  After  correction : {err_after:.6f}")
print(f"  Reduction         : {err_before - err_after:.6f}  "
      f"({100.0*(err_before-err_after)/max(err_before,1e-12):.1f}%)")
print("=" * 60)

# Plot corrected BRS (same slice as Sections 5)
po_corr = PlotOptions(
    do_plot=True,
    plot_type="set",
    plotDims=[0, 1, 3],
    slicesCut=[10],
    save_fig=True,
    filename="brs_4d_corrected.png",
)
visualize_plots(V_corrected, g_4d, po_corr)