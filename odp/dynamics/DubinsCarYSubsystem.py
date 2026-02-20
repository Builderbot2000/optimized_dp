import heterocl as hcl
import numpy as np
import math

"""
3D Y-VELOCITY-HEADING SUBSYSTEM OF THE 4D DUBINS CAR

Subsystem 2 of the overlapping decomposition of the 4D Dubins Car
  (x, y, v, theta) into two 3D subsystems.

State space:  z_y = (y, v, theta)
Control:      u = (a, omega)
Disturbance:  d_y  (additive noise on y)

Continuous-time dynamics:
  y_dot     = v * sin(theta) + d_y
  v_dot     = a
  theta_dot = omega
"""


class DubinsCarYSubsystem:
    def __init__(
        self,
        x=[0, 0, 0],
        uMin=[-1.5, -math.pi / 18],
        uMax=[1.5, math.pi / 18],
        dMin=[-0.25],
        dMax=[0.25],
        uMode="min",
        dMode="max",
    ):
        """3D Y–velocity–heading subsystem of the 4D Dubins Car.

        States (in order):
            state[0] : y position
            state[1] : speed v
            state[2] : heading theta

        Controls (in order):
            u[0] : acceleration a
            u[1] : turn rate omega

        Disturbance:
            d[0] : additive noise on y (d_y)

        Args:
            x (list):    Initial state [y, v, theta].
            uMin (list): Lower bounds of control [a_min, omega_min].
            uMax (list): Upper bounds of control [a_max, omega_max].
            dMin (list): Lower bound of disturbance [d_y_min].
            dMax (list): Upper bound of disturbance [d_y_max].
            uMode (str): "min" (reach goal) or "max" (avoid set).
            dMode (str): Opposite of uMode ("max" or "min").
        """
        self.x = x
        self.uMin = uMin
        self.uMax = uMax
        self.dMin = dMin
        self.dMax = dMax
        assert uMode in ["min", "max"]
        self.uMode = uMode
        if uMode == "min":
            assert dMode == "max"
        else:
            assert dMode == "min"
        self.dMode = dMode

    # ------------------------------------------------------------------
    # HeteroCL interface (used by HJSolver)
    # ------------------------------------------------------------------

    def opt_ctrl(self, t, state, spat_deriv):
        """Compute optimal control in the HeteroCL graph.

        Optimal control derivation (Hamiltonian maximised / minimised):
          H contains:  a * p_v  +  omega * p_theta
          =>  a*     = argopt_a  { a * p_v }
          =>  omega* = argopt_w  { omega * p_theta }

        spat_deriv indices (match state order):
            spat_deriv[0] : dV/dy
            spat_deriv[1] : dV/dv
            spat_deriv[2] : dV/dtheta
        """
        opt_a = hcl.scalar(self.uMax[0], "opt_a")
        opt_w = hcl.scalar(self.uMax[1], "opt_w")
        in3   = hcl.scalar(0, "in3")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[1] > 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(spat_deriv[2] > 0):
                opt_w[0] = self.uMin[1]
        else:  # "max"
            with hcl.if_(spat_deriv[1] < 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(spat_deriv[2] < 0):
                opt_w[0] = self.uMin[1]

        return (opt_a[0], opt_w[0], in3[0])

    def opt_dstb(self, t, state, spat_deriv):
        """Compute optimal disturbance in the HeteroCL graph.

        H contains:  d_y * p_y
          =>  d_y* = argopt_dy { d_y * p_y }

        spat_deriv[0] : dV/dy
        """
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")   # unused padding
        d3 = hcl.scalar(0, "d3")   # unused padding

        if self.dMode == "max":
            with hcl.if_(spat_deriv[0] > 0):
                d1[0] = self.dMax[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMin[0]
        else:  # "min"
            with hcl.if_(spat_deriv[0] > 0):
                d1[0] = self.dMin[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMax[0]

        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        """Continuous-time dynamics in the HeteroCL graph.

          y_dot     = v * sin(theta) + d_y
          v_dot     = a
          theta_dot = omega
        """
        y_dot     = hcl.scalar(0, "y_dot")
        v_dot     = hcl.scalar(0, "v_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        y_dot[0]     = state[1] * hcl.sin(state[2]) + dOpt[0]
        v_dot[0]     = uOpt[0]
        theta_dot[0] = uOpt[1]

        return (y_dot[0], v_dot[0], theta_dot[0])

    # ------------------------------------------------------------------
    # Pure-Python helpers (usable after value function is obtained)
    # ------------------------------------------------------------------

    def optCtrl_inPython(self, spat_deriv):
        """Compute optimal control in plain Python.

        Args:
            spat_deriv (tuple): (dV/dy, dV/dv, dV/dtheta)

        Returns:
            tuple: (opt_a, opt_omega)
        """
        opt_a = self.uMax[0]
        opt_w = self.uMax[1]

        if self.uMode == "min":
            if spat_deriv[1] > 0:
                opt_a = self.uMin[0]
            if spat_deriv[2] > 0:
                opt_w = self.uMin[1]
        else:
            if spat_deriv[1] < 0:
                opt_a = self.uMin[0]
            if spat_deriv[2] < 0:
                opt_w = self.uMin[1]

        return opt_a, opt_w

    def dynamics_inPython(self, state, action, d_y=0.0):
        """Compute continuous-time derivatives in plain Python.

        Args:
            state  (array-like): [y, v, theta]
            action (array-like): [a, omega]
            d_y    (float):      y-disturbance value

        Returns:
            tuple: (y_dot, v_dot, theta_dot)
        """
        y_dot     = state[1] * np.sin(state[2]) + d_y
        v_dot     = action[0]
        theta_dot = action[1]
        return (y_dot, v_dot, theta_dot)
