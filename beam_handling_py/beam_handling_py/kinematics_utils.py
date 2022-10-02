#!/usr/bin/env python3

import numpy as np
import casadi as cs

from beam_handling_py.panda import Limits


class VelocityLevelIKSolver:
    """ A class for solving inverse kinematics 
    at the velocity level
    """

    def __init__(self, num_solver='qpoases', ignore_velocity_constraints=False) -> None:
        # Load functions for computing ee velocity
        self.eval_vee = cs.Function.load(
            'beam_insertion_py/casadi_fcns/eval_vee.casadi')
        self.eval_Jbeam = cs.Function.load(
            'beam_insertion_py/casadi_fcns/eval_Jpend.casadi')

        # Formulating the problem
        dq = cs.SX.sym('dq', 7)
        q = cs.SX.sym('q', 7)
        vee = cs.SX.sym('vee', 6)
        g1 = vee - self.eval_Jbeam(q) @ dq
        qp = {'x': dq, 'f': 0.5*dq.T @ dq, 'g': g1, 'p': cs.vertcat(vee, q)}

        # Create a solver
        if num_solver == 'qpoases':
            self.solver = cs.qpsol('S', num_solver, qp, {
                                   'printLevel': 'none', 'print_time': 0})
        else:
            self.solver = cs.nlpsol('S', num_solver, qp, {
                                    'ipopt.print_level': 0, 'print_time': 0})

        # Get velocity limits
        limits = Limits()
        self.dq_max = limits.dq_max
        self.ddq_max = limits.ddq_max

        # Store the previous value of the joint velocity for warm start
        self.dq_prev = np.zeros((7, 1))

        # Bounds on constraints and optimization variable
        if ignore_velocity_constraints:
            self.lbx = -cs.inf
            self.ubx = cs.inf
        else:
            self.lbx = -self.dq_max
            self.ubx = self.dq_max
        self.lbg = np.zeros(6)
        self.ubg = np.zeros(6)

    def solveIK(self, vee, q, dq0=None):
        """Solves the inverse kinematics

        :parameter vee: end-effector velocity
        :parameter q: joint position
        :parameter dq0: an initial guess for joint velocities

        :return: joint velocities
        """
        x0 = dq0 if dq0 is not None else self.dq_prev

        p = cs.vertcat(vee, q)
        sol = self.solver(x0=x0, p=p, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg)
        dq = np.array(sol['x'])
        self.dq_prev = dq
        return dq


def rotm2axang(rot):
    t_ = np.array([rot[2, 1] - rot[1, 2],
                   rot[0, 2] - rot[2, 0],
                   rot[1, 0] - rot[0, 1]], ndmin=2).T
    c_alpha = 0.5*(np.sum(np.diag(rot)) - 1)
    s_alpha = 0.5*np.linalg.norm(t_)
    alpha = np.arctan2(s_alpha, c_alpha)
    if s_alpha < 1e-3:
        sign_t = 2*(t_ >= 0) - 1
        u = sign_t*np.sqrt((np.diag(rot) - c_alpha) /
                           (1 - c_alpha)).reshape(3, 1)
    else:
        u = 0.5/s_alpha*t_
    return u, alpha


def skew(x):
    """
    :return: a numpy skew symmetric matrix 
    """
    return np.array([[0, -x[2, 0], x[1, 0]],
                     [x[2, 0], 0, -x[0, 0]],
                     [-x[1, 0], x[0, 0], 0]])


def rot2rpy(R):
    """ Convert a rotation matrix to an RPY representation. The solution
    range is (-pi/2, pi/2). R = Rz(phi) Ry(theta) Rx(psi)
    """
    phi = np.arctan2(R[1, 0], R[0, 0])
    theta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    psi = np.arctan2(R[2, 1], R[2, 2])
    return np.array([psi, theta, phi])


def rot2zyz(R):
    """ Convert a rotation matrix to a Euler ZYZ angles
    R = Rz(phi) Ry'(theta) Rz''(psi)
    """
    if (np.allclose(np.abs(R[0, 2]), 0., rtol=1e-3, atol=1e-3) or
            np.allclose(np.abs(R[1, 2]), 0., rtol=1e-3, atol=1e-3)):
        raise RuntimeWarning

    phi = np.arctan2(R[1, 2], R[0, 2])
    theta = np.arctan2(np.sqrt(R[0, 2]**2 + R[1, 2]**2), R[2, 2])
    psi = np.arctan2(R[2, 1], -R[2, 0])
    return np.array([phi, theta, psi])


def axang2rotm(u, alpha):
    return np.cos(alpha)*np.eye(3) + np.sin(alpha)*skew(u) + \
        (1 - np.cos(alpha))*np.outer(u, u)


def RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix
    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs
    """
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]


def rotz(q):
    c = np.cos(q)
    s = np.sin(q)
    return np.array([[c, -s, 0.],
                     [s, c, 0.],
                     [0., 0., 1.]])


def rotz_casadi(q):
    return cs.SX(rotz(q))


def roty(q):
    c = np.cos(q)
    s = np.sin(q)
    return np.array([[c, 0., s],
                     [0., 1., 0.],
                     [-s, 0., c]])


def rotx(q):
    c = np.cos(q)
    s = np.sin(q)
    return np.array([[1., 0., 0.],
                     [0., c, -s],
                     [0., s, c]])


def drotz(q):
    c = np.cos(q)
    s = np.sin(q)
    return np.array(([-s, -c, 0.],
                     [c, -s, 0.],
                     [0., 0., 1.]))


def drotz_casadi(q):
    return cs.SX(drotz(q))


def estimate_accelerations(dq, ts):
    """ Numerically estimates joint accelerations
    """
    ddq = np.zeros_like(dq)
    dq_prev = dq[0, :]
    for k, dqk in enumerate(dq[1:, :]):
        ddq[k+1, :] = (dqk - dq_prev)/ts
        dq_prev = dqk

    return ddq
