# 3D Overlapping Subsystem Decomposition for 4D Dubin's Car in Hamilton--Jacobi Reachability

**Kevin Tang**\
February 2026

------------------------------------------------------------------------

## 1. Introduction

In this document, we define a 4D Dubin's Car system and its
decomposition into two overlapping 3D subsystems suitable for
Hamilton--Jacobi reachability analysis.

------------------------------------------------------------------------

## 2. Base 4D System

### State space

x = (x, y, v, θ)

### Control input

u = (a, ω)

### Control bounds

a ∈ \[a_min, a_max\]\
ω ∈ \[ω_min, ω_max\]

### Disturbance

d = (d_x, d_y)\
d_x ∈ D_x, d_y ∈ D_y

### Continuous-time dynamics

ẋ = v cos θ + d_x\
ẏ = v sin θ + d_y\
v̇ = a\
θ̇ = ω

------------------------------------------------------------------------

## 3. Subsystem 1: X--Velocity--Heading System

### State space

z_x = (x, v, θ)

### Control input

u_x = (a, ω)

### Control bounds

a ∈ \[a_min, a_max\]\
ω ∈ \[ω_min, ω_max\]

### Disturbance

d_x ∈ D_x

### Continuous-time dynamics

ẋ = v cos θ + d_x\
v̇ = a\
θ̇ = ω

------------------------------------------------------------------------

## 4. Subsystem 2: Y--Velocity--Heading System

### State space

z_y = (y, v, θ)

### Control input

u_y = (a, ω)

### Control bounds

a ∈ \[a_min, a_max\]\
ω ∈ \[ω_min, ω_max\]

### Disturbance

d_y ∈ D_y

### Continuous-time dynamics

ẏ = v sin θ + d_y\
v̇ = a\
θ̇ = ω

------------------------------------------------------------------------

## 5. Subsystem Coupling for Reachability

The overlapping velocity--heading states (v, θ) couple the two
subsystems, ensuring that each subsystem respects the same speed and
heading bounds. For Hamilton--Jacobi reachability, (a, ω) are treated as
bounded controls, which yields a sound decomposition of the original 4D
system into two overlapping 3D systems.

------------------------------------------------------------------------

## 6. Reconstructing the 4D Grid from 3D Subsystems

The 4D backward reachable set (BRS) can be obtained by first computing
the BRS for each 3D subsystem and then combining them conservatively.

Let:

-   V_x(x, v, θ) be the BRS for the X-subsystem\
-   V_y(y, v, θ) be the BRS for the Y-subsystem

### Step 1: Compute the 3D BRS grids

-   Compute V_x(x, v, θ) for the X-subsystem using Optimized DP on a 3D
    grid over (x, v, θ).
-   Compute V_y(y, v, θ) for the Y-subsystem using Optimized DP on a 3D
    grid over (y, v, θ).

### Step 2: Define the 4D grid

Construct a 4D grid over (x, y, v, θ) with the same resolution as the 3D
grids:

(x_i, y_j, v_k, θ_l)\
i = 1...N_x, j = 1...N_y, k = 1...N_v, l = 1...N_θ

### Step 3: Combine 3D BRS values into 4D

For each point on the 4D grid, define the 4D BRS value as:

V_4(x_i, y_j, v_k, θ_l) = max( V_x(x_i, v_k, θ_l), V_y(y_j, v_k, θ_l) )
