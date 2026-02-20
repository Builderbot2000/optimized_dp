# Experiment: 3D Overlapping Subsystem Decomposition for 4D Dubins Car BRS

**Kevin Tang** — February 2026

---

## Overview

This experiment validates the overlapping 3D decomposition of a 4D Dubins Car system
for Hamilton--Jacobi reachability analysis.  The theoretical design is specified in
[`decomposition_design_4D_to_3D.md`](decomposition_design_4D_to_3D.md).

Two backward reachable sets (BRS) are computed and held in memory for comparison:

| Variable | Description | Shape |
|---|---|---|
| `last_time_step_4d` | Direct 4D HJ solve (ground truth) | `(60, 60, 20, 36)` |
| `V_4d` | Reconstructed 4D BRS from two 3D subsystem solves | `(60, 60, 20, 36)` |

The reconstructed BRS is a conservative outer approximation: it is expected to contain
(be larger than or equal to) the direct 4D BRS at every grid point.

---

## Base 4D System (`DubinsCar4D2`)

| Property | Value |
|---|---|
| State | $(x,\ y,\ v,\ \theta)$ |
| Control | $(a,\ \omega)$ |
| Disturbance | $(d_x,\ d_y)$ |
| Dynamics | $\dot{x} = v\cos\theta + d_x,\ \dot{y} = v\sin\theta + d_y,\ \dot{v} = a,\ \dot{\theta} = \omega$ |
| Control bounds | $a \in [-1.5,\ 1.5]$, $\omega \in [-\pi/18,\ \pi/18]$ |
| Disturbance bounds | $d_x,\ d_y \in [-0.25,\ 0.25]$ |
| Mode | `uMode="max"`, `dMode="min"` (avoid set) |

---

## Decomposition

### Subsystem 1 — X–velocity–heading (`DubinsCarXSubsystem`)

| Property | Value |
|---|---|
| State | $(x,\ v,\ \theta)$ |
| Dynamics | $\dot{x} = v\cos\theta + d_x,\ \dot{v} = a,\ \dot{\theta} = \omega$ |
| Disturbance | $d_x \in [-0.25,\ 0.25]$ |
| Implementation | [`odp/dynamics/DubinsCarXSubsystem.py`](../../odp/dynamics/DubinsCarXSubsystem.py) |

### Subsystem 2 — Y–velocity–heading (`DubinsCarYSubsystem`)

| Property | Value |
|---|---|
| State | $(y,\ v,\ \theta)$ |
| Dynamics | $\dot{y} = v\sin\theta + d_y,\ \dot{v} = a,\ \dot{\theta} = \omega$ |
| Disturbance | $d_y \in [-0.25,\ 0.25]$ |
| Implementation | [`odp/dynamics/DubinsCarYSubsystem.py`](../../odp/dynamics/DubinsCarYSubsystem.py) |

### Coupling

The shared states $(v, \theta)$ and shared control $(a, \omega)$ couple the two subsystems.
Each 3D solve respects the same speed and heading bounds as the full 4D system, making the
decomposition sound.

---

## Solver Parameters

| Parameter | Value |
|---|---|
| Look-back length | 1.5 s |
| Time step | 0.05 s |
| Time steps | 31 |
| Computation method | `minVWithV0` (BRT) |
| Spatial accuracy | `"medium"` |

---

## Grid Specifications

### 4D grid (`g_4d`)

| Dimension | Min | Max | Points | Periodic |
|---|---|---|---|---|
| $x$ | −3.0 | 3.0 | 60 | No |
| $y$ | −1.0 | 4.0 | 60 | No |
| $v$ | 0.0 | 4.0 | 20 | No |
| $\theta$ | $-\pi$ | $\pi$ | 36 | Yes |

### X-subsystem grid (`g_x`)

| Dimension | Min | Max | Points | Periodic |
|---|---|---|---|---|
| $x$ | −3.0 | 3.0 | 60 | No |
| $v$ | 0.0 | 4.0 | 20 | No |
| $\theta$ | $-\pi$ | $\pi$ | 36 | Yes |

### Y-subsystem grid (`g_y`)

| Dimension | Min | Max | Points | Periodic |
|---|---|---|---|---|
| $y$ | −1.0 | 4.0 | 60 | No |
| $v$ | 0.0 | 4.0 | 20 | No |
| $\theta$ | $-\pi$ | $\pi$ | 36 | Yes |

---

## Target Sets

| Solve | Shape | Ignored dims | Centre | Radius |
|---|---|---|---|---|
| 4D direct | Cylinder in $(x, y)$ | $v,\ \theta$ | $(0,\ 2)$ | 0.8 |
| X-subsystem | Cylinder in $x$ | $v,\ \theta$ | $x = 0$ | 0.8 |
| Y-subsystem | Cylinder in $y$ | $v,\ \theta$ | $y = 2$ | 0.8 |

---

## 4D BRS Reconstruction

After computing $V_x(x, v, \theta)$ and $V_y(y, v, \theta)$, the 4D BRS is reconstructed
by pointwise max over the two subsystem value functions (Section 6 of the design doc):

$$V_{4D}(x_i,\ y_j,\ v_k,\ \theta_l) = \max\!\bigl(V_x(x_i,\ v_k,\ \theta_l),\ V_y(y_j,\ v_k,\ \theta_l)\bigr)$$

In NumPy this is implemented via broadcasting:

```python
V_4d = np.maximum(
    V_x[:, None, :, :],   # (60,  1, 20, 36)
    V_y[None, :, :, :],   # ( 1, 60, 20, 36)
)                          # -> (60, 60, 20, 36)
```

---

## Visualization

Both BRS results are plotted as zero-sublevel-set isosurfaces in the $(x, y, \theta)$
subspace at $v$-index 10, consistent with the `dubins_4d_avoid.py` example.

| Output file | Description |
|---|---|
| `brs_4d_direct.png` | Animated BRT from the direct 4D solve (one frame per time step) |
| `brs_4d_decomposed.png` | Static BRS from the reconstructed 4D value function |

---

## Files

| File | Role |
|---|---|
| [`dubins_4d_to_3d.py`](dubins_4d_to_3d.py) | Main experiment script |
| [`decomposition_design_4D_to_3D.md`](decomposition_design_4D_to_3D.md) | Theoretical design and decomposition derivation |
| [`odp/dynamics/DubinsCarXSubsystem.py`](../../odp/dynamics/DubinsCarXSubsystem.py) | X–v–θ subsystem dynamics class |
| [`odp/dynamics/DubinsCarYSubsystem.py`](../../odp/dynamics/DubinsCarYSubsystem.py) | Y–v–θ subsystem dynamics class |

---

## Running the Experiment

From the workspace root:

```bash
python experiments/dubins_4d_to_3d/dubins_4d_to_3d.py
```

Expected terminal output:

```
4D direct BRS shape:          (60, 60, 20, 36)
Reconstructed 4D BRS shape:  (60, 60, 20, 36)
Done. Results held in memory as `last_time_step_4d` and `V_4d`.
```
