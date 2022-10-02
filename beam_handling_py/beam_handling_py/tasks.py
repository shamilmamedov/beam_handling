#!/usr/bin/env python3

import numpy as np
from beam_handling_py.models import SymbolicSetupKinematics


class P2PMotionTask:
    """ Generic class for point-to-point motion
    """

    def __init__(self, q_t0, pb_t0, Rb_t0, pb_tf, Rb_tf) -> None:
        n_dof = 7
        self.dq_t0 = np.zeros((n_dof, 1))
        self.dq_tf = np.zeros((n_dof, 1))
        self.q_t0 = q_t0
        self.pb_t0 = pb_t0
        self.Rb_t0 = Rb_t0
        self.pb_tf = pb_tf
        self.Rb_tf = Rb_tf

    def __str__(self) -> str:
        if np.array_equal(self.Rb_t0, self.Rb_tf):
            return 'Translational point-to-point motion task'
        else:
            return 'Spatial point-to-point motion task'


class TransP2PMotionTask(P2PMotionTask):
    """ Translational point to point motion
    """

    def __init__(self, q_t0: np.ndarray, delta_pb: np.ndarray) -> None:
        # Instantiate setup kinematics class
        symkin = SymbolicSetupKinematics()

        # Forward kinematics to ee orientation and position at initial position
        pb_t0 = np.array(symkin.eval_pb(q_t0))
        Rb_t0 = np.array(symkin.eval_Rb(q_t0))
        pb_tf = pb_t0 + delta_pb

        # Orientations in the beginning and at the end are the same
        R_tf = np.copy(Rb_t0)

        # Initialize parent class
        super().__init__(q_t0, pb_t0, Rb_t0, pb_tf, R_tf)


def ICRA_task_1():
    # Initial configuration
    q_t0 = np.array([[-np.pi/2, -np.pi/6, 0., -2*np.pi/3,
                      -np.pi/2, np.pi/2, np.pi/4]]).T

    # Position increment in global frame
    delta_pb = np.array([[-0.2, 0., 0.]]).T
    return TransP2PMotionTask(q_t0, delta_pb)


def ICRA_task_2():
    # Initial configuration
    q_t0 = np.array([[-np.pi/2, -np.pi/6, 0., -2*np.pi/3,
                      0., np.pi/2, np.pi/4]]).T

    # Position increment in global frame
    delta_pb = np.array([[0.2, 0., -0.2]]).T
    return TransP2PMotionTask(q_t0, delta_pb)


def ICRA_task_3():
    # Initial configuration
    q_t0 = np.array([[-np.pi/2, -np.pi/6, 0., -2*np.pi/3,
                      np.pi/2, np.pi/2, np.pi/4]]).T

    # Position increment in global frame
    delta_pb = np.array([[0.2, 0., -0.2]]).T

    # Instantiate setup kinematics class
    symkin = SymbolicSetupKinematics()

    # Initial {b} frame position and orientations
    pb_t0 = np.array(symkin.eval_pb(q_t0))
    Rb_t0 = np.array(symkin.eval_Rb(q_t0))

    # Final {b} frame positions
    pb_tf = pb_t0 + delta_pb

    # Final {b} frame orientation
    q_tmp = np.array([[-np.pi/2, -np.pi/6, 0., -2*np.pi/3,
                       0., np.pi/2, 3*np.pi/4]]).T
    Rb_tf = np.array(symkin.eval_Rb(q_tmp))
    return P2PMotionTask(q_t0, pb_t0, Rb_t0, pb_tf, Rb_tf)


def ICRA_tasks(name: str):
    """ Aggregates all specified tasks. 

    :parameter name: name of the task according to the paper
    """
    if name == 'T1':
        return ICRA_task_1()
    elif name == 'T2':
        return ICRA_task_2()
    elif name == 'T3':
        return ICRA_task_3()
    else:
        raise ValueError


if __name__ == "__main__":
    t = ICRA_task_2()
    print(type(t.pb_tf))
