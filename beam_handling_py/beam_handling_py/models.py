#!/usr/bin/env python3

import numpy as np
import casadi as cs

from beam_handling_py.kinematics_utils import drotz_casadi, rotz_casadi


class SymbolicSetupKinematics:
    """ Class implements/contains functions for computing forward
    kinematics of the setup: beam frame position, and orientation,
    velocities and accelerations. It loads casadi functions
    generated from Pinocchio.
    """

    def __init__(self) -> None:
        """
        """
        fcns_dir = 'beam_handling_py/casadi_fcns/'

        self.eval_pb = cs.Function.load(fcns_dir + f'eval_ppend.casadi')
        self.eval_Rb = cs.Function.load(fcns_dir + f'eval_Rpend.casadi')
        self.eval_vb = cs.Function.load(fcns_dir + 'eval_vpend.casadi')
        self.eval_ab = cs.Function.load(fcns_dir + 'eval_apend.casadi')
        self.eval_Jb = cs.Function.load(fcns_dir + 'eval_Jpend.casadi')
        self.eval_dJb = cs.Function.load(fcns_dir + 'eval_dJpend.casadi')

        self.nq_arm = 7
        self.nq_beam = 1


class SymbolicModel:
    """ Implements ordinary differential equation (ODE) in symbolic
    form using Casadi. ODE is represented in the following form
    dx = f(x, u), here dx represents time derivative of x

    Class variables:
        nx -- number of states
        nu -- number of controls
        np -- number of parameters
        x -- (vector) of symbolic variables for states
        u -- (vector) of symbolic variables for controls
        p -- (vector) of symbolic variables for parameters (static, constant over time)
        rhs -- symbolic expression of the right hand side of ODE: f(x, u)
        ode -- a casadi function for rhs
    """

    def __init__(self, x, u, rhs, p=None) -> None:
        """
        :parameter x: [nx x 1] symbolic variable representing states
        :parameter u: [nu x 1] symbolic variable representing control inputs        
        :parameter p: [np x 1] symbolic variable representing parameters
        :parameter rhs: [nx x 1] symbolic expressiion representing right hand side
                        of the ode -- f(x, u)
        :parameter ndq_arm: number of velocity variables of the arm. It can be 7 for
                            kinematic model or 14 for first order model
        """
        # Process inputs
        self.nx = x.shape[0]
        self.nu = u.shape[0]
        self.x = x
        self.u = u
        self.rhs = rhs

        # Proccess a vector of parameters
        if p is not None:
            self.p, self.np = p, p.shape[0]
        else:
            self.p, self.np = [], 0

        # Define a vector of inputs of the ode
        ode_in = [self.x, self.u, self.p]
        ode_in_labels = ['x', 'u', 'p']

        # Create a function for ode
        self.ode = cs.Function('ode', ode_in, [rhs],
                               ode_in_labels, ['dx'])

    def __str__(self) -> str:
        t_ = (f"Symbolic dynamic model with nx={self.nx}, nu={self.nu}, " +
              f"and np={self.np}")
        return t_


def get_beam_params(type: str = "estimated") -> np.ndarray:
    """ Returns beam parameters: natural frequency, damping ratio and length.

    :parameter type: specifies params
    """
    # In pendulum approximation we need to know only the natural
    # frequency along Z axis, the model takes cares of the change of
    # natural frequency when the orientation of the end-effector is
    # different from Z using L as an additional parameter

    if type == "estimated":
        wn, zeta, L = 18.42, 0.004, 0.34

    elif type == "analytical":
        # For damping I choose the lowest value
        wn, zeta, L = 18.44, 0.005, 0.52
    else:
        raise ValueError

    return np.array([wn, zeta, L])


def arm_kinematic_model():
    """ ode for robot dynamics where it is assumed that each joint is
    a double integrator: kinematic robot model
    """
    # Variables for defning dynamics
    q = cs.SX.sym('q', 7)
    dq = cs.SX.sym('dq', 7)
    u = cs.SX.sym('u', 7)

    rhs = cs.vertcat(dq, u)
    x = cs.vertcat(q, dq)
    model = SymbolicModel(x, u, rhs)
    return model


def beam_dynamics(q: cs.SX.sym, dq: cs.SX.sym, u: cs.SX.sym, p: cs.SX.sym,
                  symkin: "SymbolicSetupKinematics"):
    """ Return a symbolic expressions for beam dynamuics approximated 
    as a spring-mass -damper system

    :parameter q: a symbolic joint state vector
    :parameter dq: a symbolic joint velocity vectyor
    :parameter u: a symbolic input (joitn accleration vector)
    :parameter p: [wn, zeta, L] a syumbolic vector of beam (pendulum) params
    :parameter symkin: symbolic kinematics of the setup

    :return: a list of beam position, a list of beam velocity and a 
            a list of beam acceleration
    """

    # Find orientation, linear velocity and acceleration of the ee
    R = symkin.eval_Rb(q)
    dpee = symkin.eval_vb(q, dq)
    ddpee = symkin.eval_ab(q, dq, u)

    dpee_v, dpee_w = dpee[:3], dpee[3:]
    ddpee_v = ddpee[:3]
    dwee = ddpee[3:]

    # Gravity vector and parameters
    g0 = 9.81
    g = np.array([[0., 0., -g0]]).T
    wn, zeta, L = p[0], p[1], p[2]

    theta = cs.SX.sym(f'theta', 1)
    dtheta = cs.SX.sym(f'dtheta', 1)
    S_w = cs.skew(dpee_w)
    S_dw = cs.skew(dwee)

    k = np.array([[1., 0., 0.]]).T
    ddtheta = (-2*zeta*wn*dtheta - wn**2*theta + 1/L*k.T @ drotz_casadi(theta).T @ R.T @ (g - ddpee_v)
               - k.T @ drotz_casadi(theta).T @ R.T @ S_w @ S_w @ R @ rotz_casadi(theta) @ k
               - k.T @ drotz_casadi(theta).T @ R.T @ S_dw @ R @ rotz_casadi(theta) @ k)

    return theta, dtheta, ddtheta


def beam_rest_position(R, p: np.ndarray):
    """ Calculates rest position of the beam (spring-mass-damper equivalent)

    :parameter R: orientation of robot
    :parameter p: vector of parameters [wn, zeta, L]

    :return theta: position of the mass
    """
    # Parse parameters
    wn, zeta, L = p
    g = np.array([[0., 0., -9.81]]).T

    # Solve root finding problem
    k = np.array([[1., 0., 0.]]).T
    theta_sym = cs.SX.sym('theta')
    g_sym = 1/L*k.T @ drotz_casadi(theta_sym).T @ R.T @ g - wn**2*theta_sym
    g = cs.Function('g', [theta_sym], [g_sym])
    G = cs.rootfinder('G', 'newton', g)
    theta = np.array(G(0.)).squeeze()

    return theta


def setup_kinematic_model(symkin: "SymbolicSetupKinematics"):
    """ ode for joints of the robot as double integrator + 
    spring mass damper system representing beam 

    :parameter approximation: beam approximation = ['smd', 'pendulum']
    :parameter formulation: formulation of the dynamics (# of dofs and axis of motion)
    :parameter beam_params: specifies beam parameters, could be string,
                            a list of strings or tuple with parameter values (wn, zeta)
    :parameter beam_axis: defines orientation of the beam (Z-horizontal, Xup, Xdown-vertical)
    """
    # Create casadi variables for Panda arm to describe dynamics
    q = cs.SX.sym('q', 7)
    dq = cs.SX.sym('dq', 7)
    u = cs.SX.sym('u', 7)

    # Beam parameters
    wn = cs.SX.sym('wn')
    zeta = cs.SX.sym('zeta')
    L = cs.SX.sym('L')
    p = cs.vertcat(wn, zeta, L)

    # Beam dynamics
    theta, dtheta, ddtheta = beam_dynamics(q, dq, u, p, symkin)

    # Compose state vector, right-hand side and create a model
    x = cs.vertcat(q, theta, dq, dtheta)
    rhs = cs.vertcat(dq, dtheta, u, ddtheta)
    model = SymbolicModel(x, u, rhs, p=p)
    return model


if __name__ == "__main__":
    arm_model = arm_kinematic_model()

    symkin = SymbolicSetupKinematics('pend')
    setup_model = setup_kinematic_model(symkin)
    print(setup_model)

    p = np.array([18.42, 0.005, 0.34])
    q = np.array(
        [[-np.pi/2, -np.pi/6, 0., -2*np.pi/3, 0., np.pi/2, np.pi/4]]).T
    R = symkin.eval_Rb(q)
    theta_eq = beam_rest_position(R, p)
    print(f"Equilibrium positionm = {theta_eq:.4f}")
