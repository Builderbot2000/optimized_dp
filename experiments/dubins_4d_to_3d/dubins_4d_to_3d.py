import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import CylinderShape
from odp.Plots import PlotOptions, visualize_plots
from odp.solver import HJSolver
from odp.dynamics import DubinsCar4D2, DubinsCarXSubsystem, DubinsCarYSubsystem

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