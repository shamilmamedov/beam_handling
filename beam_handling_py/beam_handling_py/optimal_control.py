#!/usr/bin/env python3

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from numpy.matlib import repmat

import beam_handling_py.panda as panda
from beam_handling_py.simulation import symbolic_RK4, simulate_system
from beam_handling_py.models import (SymbolicModel, get_beam_params, arm_kinematic_model, 
                                     setup_kinematic_model, SymbolicSetupKinematics, 
                                     beam_rest_position)
from beam_handling_py.poly_trajectory_planning import Poly5Trajectory
from beam_handling_py.kinematics_utils import RpToTrans
from beam_handling_py.visualizer import SetupVisualizer
from beam_handling_py.tasks import ICRA_tasks


class OCPParameters():
    """ Optimal control parameters for joint space optimal motion planning
    """

    def __init__(self, ts: float = 0.01, N: int = 56, rho: float = 10.,
                 suppress_vibrations: bool = False, R=None, Q=None, W=None) -> None:
        """
        :parameter ts: sampling time
        :parameter N: number of samples -> T = ts*N
        :parameter rho: jerk penalty
        :parameter R: control penalty (acceleration penalty)
        :parameter Q: states penalty
        :parameter W: ILC penalty with previous iteration
        """
        # Parameters of the problem
        self.N = N
        self.ts = ts
        self.n_rk4_steps = 2
        self.suppress_vibrations = suppress_vibrations

        # Structure of the problem
        self.diagonalize_jacobian = True

        # Weighting functions of the cost function
        self.rho = rho
        self.R = np.diag([1., 1., 1., 1., 5., 5., 5.]) if R is None else R
        self.Q = np.diag([0.1]*7 + [1]*7) if Q is None else Q
        self.W = np.diag([0]*7) if W is None else W

        # Get limits for panda
        # NOTE probably they should be somewhere else
        limits = panda.Limits()

        self.q_min = limits.q_min
        self.q_max = limits.q_max
        self.dq_max = limits.dq_max
        self.u_max = limits.ddq_max
        self.du_max = limits.dddq_max


class OptimalControlProblem():
    """ Formulates and solves optimal control problem
    using Opti stack from Casadi
    """

    def __init__(self, model: SymbolicModel, ocp_params: OCPParameters, boundary_constr, symkin) -> None:
        """ Initializes class instance. 

        :parameter model: symbolic ODE representing dynamics of the system
        :type ode: symbolic_RK4
        :parameter ocp_params: an object that contains ocp parameters 
        :parameter boundary_constr: a dictionary defining boundary constraints
        :parameter symkin: symbolic kinematics of the setup
        """
        # Helper parameters
        self.nq_arm = symkin.nq_arm  # number of positional states of the arm
        self.ndq_arm = symkin.nq_arm # number of velocity states of the arm
        # number of posistion states
        self.nq = self.nq_arm + (model.nx - self.nq_arm - self.ndq_arm)//2

        # Symbolic kinematic model
        self.symkin = symkin

        # OCP parameters
        self.params = ocp_params

        # System dynamics
        self.model = model
        self.F_rk4 = symbolic_RK4(self.model.x, self.model.u, self.model.p,
                                  self.model.ode, n=self.params.n_rk4_steps)

        # space where control is conducted, can be cartesian or joint
        self.x_t0 = boundary_constr["x_t0"]
        self.pb_tf = boundary_constr["pb_tf"]
        self.Rb_tf = boundary_constr["Rb_tf"]
        
        # casadi function for computing rotation error
        self.rot_err = rotation_error_fcn(self.symkin.eval_Rb, self.Rb_tf)

        if self.params.suppress_vibrations:
            self.theta_tf = boundary_constr["theta_tf"]

        # OCP formulation using opti stack
        if self.params.diagonalize_jacobian:
            self.formulate_with_diag_structure()
        else:
            self.formulate()

        self.plot_jacobian_structure()

    def formulate(self):
        """ Formulates OCP using Opti stack
        """
        # Set opti environment
        self.__opti = cs.Opti()

        # Decision variables
        self.__x = self.__opti.variable(self.model.nx, self.params.N+1)
        self.__u = self.__opti.variable(self.model.nu, self.params.N)
        self.__p = self.__opti.parameter(self.model.np, 1)
        ts = self.params.ts

        # Dynamics constraints RK4 single step
        for k in range(self.params.N):
            x_next = self.F_rk4(self.__x[:, k], self.__u[:, k], self.__p, ts)
            self.__opti.subject_to(x_next == self.__x[:, k+1])

        # Path constraints
        # Control effort constraints
        self.__opti.subject_to(
            self.__opti.bounded(-self.params.u_max, self.__u, self.params.u_max))

        # Constraints on the rate of change of control effort
        for k in range(1, self.params.N):
            self.__opti.subject_to(self.__opti.bounded(
                -ts*self.params.du_max, self.__u[:, k] - self.__u[:, k-1], ts*self.params.du_max))

        # Constraints on joint positions and velocities
        self.__opti.subject_to(self.__opti.bounded(
            self.params.q_min, self.__x[:self.nq_arm, :], self.params.q_max))
        self.__opti.subject_to(self.__opti.bounded(-self.params.dq_max,
                               self.__x[self.nq:self.nq+self.nq_arm, :], self.params.dq_max))
        
        # Boundary conditions
        # State boundary conditions
        self.__opti.subject_to(self.__x[:, 0] == self.x_t0)

        # EE position and orientation constraint at final point
        pb_tf = self.symkin.eval_pb(self.__x[:self.nq_arm, -1])
        self.__opti.subject_to(pb_tf == self.pb_tf)
        self.__opti.subject_to(self.rot_err(self.__x[:self.nq_arm, -1]) == 0.)

        # Panda joint velocity contraint
        self.__opti.subject_to(self.__x[self.nq:self.nq+self.ndq_arm, -1] == 0.)

        # Beam constraints
        if self.params.suppress_vibrations:
            self.__opti.subject_to(self.__x[self.nq_arm:self.nq, -1] == self.theta_tf)
            self.__opti.subject_to(self.__x[self.nq+self.ndq_arm:, -1] == 0.)

        # Input boundary conditions
        self.__opti.subject_to(self.__u[:, [0, -1]] == 0.)

        # Objective
        # Stage cost
        objective = 0
        for k in range(self.params.N):
            # Qadratic cost on state and controls
            objective += (self.__x[:, k] - self.x_t0).T @ self.params.Q @ (self.__x[:, k] - self.x_t0) + \
                          self.__u[:, k].T @ self.params.R @ self.__u[:, k]
            # Quadratic cost on jerks
            if k > 0:
                objective += self.params.rho * (self.__u[:, k] - self.__u[:, k-1]).T @ \
                             (self.__u[:, k] - self.__u[:, k-1])

        # Terminal cost
        self.__opti.minimize(objective)

    def formulate_with_diag_structure(self):
        """ Formulate the problem in a way that the Jacobian
        has a diagonal strucure
        """
        # Set opti environment
        self.__opti = cs.Opti()

        # Decision variables
        # Create the decision variables vector
        nx = self.model.nx
        nu = self.model.nu
        N = self.params.N
        ts = self.params.ts
        w = self.__opti.variable(nx*(N+1) + nu*N)
        self.__p = self.__opti.parameter(self.model.np, 1)

        # Set the indices of the state and control variables in the w
        x_idx = np.zeros((nx, N+1), dtype=int)
        u_idx = np.zeros((nu, N), dtype=int)

        x_idx[:, 0] = range(0, nx)
        u_idx[:, 0] = range(nx, nx+nu)

        for k in range(1, N+1):
            t1_ = int(u_idx[-1, k-1])
            x_idx[:, k] = range(1 + t1_, 1 + nx + t1_)
            if k < N:
                t2_ = int(x_idx[-1, k])
                u_idx[:, k] = range(1 + t2_, 1 + nu + t2_)

        # State and control variables
        x = w[x_idx]
        u = w[u_idx]

        # Constraints
        # Initial boundary constraints
        self.__opti.subject_to(x[:, 0] == self.x_t0)
        self.__opti.subject_to(u[:, 0] == 0.)
        for k in range(N):
            # Dynamics (multiple shooting constraint)
            x_next = self.F_rk4(x[:, k], u[:, k], self.__p, ts)
            self.__opti.subject_to(x_next == x[:, k+1])

            # Control and state bounds
            qk = x[:self.nq_arm, k]
            dqk = x[self.nq:self.nq+self.nq_arm, k]
            self.__opti.subject_to(self.__opti.bounded(-self.params.u_max, 
                                   u[:, k], self.params.u_max))
            self.__opti.subject_to(self.__opti.bounded(self.params.q_min, 
                                   qk, self.params.q_max))
            self.__opti.subject_to(self.__opti.bounded(-self.params.dq_max, 
                                   dqk, self.params.dq_max))

            # Constraints on the rate of change of control effort
            if k > 0:
                self.__opti.subject_to(self.__opti.bounded(
                    -ts*self.params.du_max, u[:, k] - u[:, k-1], ts*self.params.du_max))

        # Final boundary constraints
        # EE position and orientation constraint at final point
        pb_tf = self.symkin.eval_pb(x[:self.nq_arm, -1])
        self.__opti.subject_to(pb_tf == self.pb_tf)
        self.__opti.subject_to(self.rot_err(x[:self.nq_arm, -1]) == 0.)

        # Panda joint velocity contraint
        self.__opti.subject_to(x[self.nq:self.nq+self.ndq_arm, -1] == 0.)

        # Beam constraints
        if self.params.suppress_vibrations:
            self.__opti.subject_to(x[self.nq_arm:self.nq, -1] == self.theta_tf)
            self.__opti.subject_to(x[self.nq+self.ndq_arm:, -1] == 0.)

        # Input boundary conditions
        self.__opti.subject_to(u[:, -1] == 0.)

        # Objective
        # Stage cost
        objective = 0
        for k in range(N):
            objective += (x[:, k] - self.x_t0).T @ self.params.Q @ (x[:, k] - self.x_t0) + \
                         u[:, k].T @ self.params.R @ u[:, k]
            if k > 0:
                objective += self.params.rho * \
                             (u[:, k] - u[:, k-1]).T @ (u[:, k] - u[:, k-1])

        # Terminal cost
        self.__opti.minimize(objective)

        # Save some variables for later use
        self.__w = w
        self.x_from_w = x_idx
        self.u_from_w = u_idx

    def solve(self, p: np.ndarray, x0=None, u0=None):
        """ Solves OCP with IPOP solver
        """
        # Set parameters of the system
        self._p = p
        self.__opti.set_value(self.__p, p)

        # Initial guess using polynomial trajectory
        if self.params.diagonalize_jacobian:
            # Parse states and inputs into a decision vector
            w0 = np.zeros((self.model.nx*(self.params.N+1) +
                           self.model.nu*self.params.N, 1))
            for k in range(0, self.params.N+1):
                w0[self.x_from_w[:, k], 0] = x0[:, [k]].T
                if k < self.params.N:
                    w0[self.u_from_w[:, k], 0] = u0[:, [k]].T
            # Iniitalize decision variables
            self.__opti.set_initial(self.__w, w0)
        else:
            if x0 is None:
                self.__opti.set_initial(self.__x, repmat(
                    self.x_t0.reshape(-1, 1), 1, self.params.N+1))
            else:
                self.__opti.set_initial(self.__x, x0)

            if u0 is not None:
                self.__opti.set_initial(self.__u, u0)


        # Solver settings
        p_opts = {"expand": True}  # plugin options
        s_opts = {'max_iter': 250, 'print_level': 1, 'linear_solver': 'mumps',
                  'hessian_approximation': 'limited-memory'}  # solver options
        self.__opti.solver("ipopt", p_opts, s_opts)

        # Solve OCP
        sol = self.__opti.solve()
        if self.params.diagonalize_jacobian:
            w_opt = sol.value(self.__w)
            x_opt = w_opt[self.x_from_w].T
            u_opt = w_opt[self.u_from_w].T
        else:
            u_opt = sol.value(self.__u).T.reshape(-1, self.model.nu)
            x_opt = sol.value(self.__x).T
        t_opt = np.arange(0, self.params.N+1, 1)*self.params.ts
      
        self._t_opt = t_opt
        self._x_opt = x_opt
        self._u_opt = u_opt

        return t_opt, x_opt, u_opt

    def resample_solution(self, ts):
        """ Resamples solution of the ocp according to a given sampling time

        :param ts: sampling time
        """
        # A check!!!!!!
        np.testing.assert_equal(self.params.ts, 0.01)
        u_res = np.repeat(self._u_opt, 10, axis=0)
        x0 = self._x_opt[[0], :].T
        t_res, x_res = simulate_system(x0, u_res, self._p, ts, 
                                       self.params.N*10, self.model, 'cvodes')
        return t_res, x_res

    def save_solution(self):
        """ Saves solution of the ocp for two purposes: joint velocities to 
        execute on the robot, and positions to visualize in rviz
        """
        dq_ref_idx_i = self.nq  # initial dq_ref index
        dq_ref_idx_f = self.nq + self.nq_arm

        if self.params.ts == 0.001:
            q_ref = self._x_opt[::10, :self.nq_arm]
            dq_ref = self._x_opt[:, dq_ref_idx_i:dq_ref_idx_f]
        elif self.params.ts == 0.01:
            q_ref = self._x_opt[:, :self.nq_arm]
            t_resamp, x_resamp = self.resample_solution(0.001)
            dq_ref = x_resamp[:, dq_ref_idx_i:dq_ref_idx_f]
        else:
            raise NotImplementedError

        # pad reference with zeros
        dq_ref = np.pad(dq_ref, ((100, 100), (0, 0)), mode='constant')
        q_ref = np.pad(q_ref, ((100, 300), (0, 0)), mode='edge')
        np.savetxt('js_opt_traj.csv', dq_ref,  fmt='%.20f', delimiter=',')
        np.savetxt('js_opt_traj_4rviz.csv', q_ref, delimiter=',')

    def plot_jacobian_structure(self):
        plt.spy(cs.jacobian(self.__opti.g, self.__opti.x).sparsity())
        plt.show()

    def visualize_solution(self):
        # Options for plotting
        plot_ee_pose = False
        plot_pendulum_states = True
        plot_controls = False
        plot_joint_states = False
        plot_ee_velocity = True
        plot_ee_acceleration = True

        # process optimal solution and visualize
        # compute end-effector position and rotation error
        q_opt = self._x_opt[:, :self.nq_arm]
        dq_opt = self._x_opt[:, self.nq:self.nq+self.nq_arm]
        pee = np.zeros((self.params.N+1, 3))
        dpee = np.zeros((self.params.N+1, 6))
        ddpee = np.zeros((self.params.N+1, 6))
        e_rot = np.zeros((q_opt.shape[0], 3))
        for k in range(self.params.N+1):
            pee[[k], :] = self.symkin.eval_pb(q_opt[k, :]).T
            e_rot[[k], :] = self.rot_err(q_opt[k, :]).T
            dpee[[k], :] = self.symkin.eval_vb(q_opt[k, :], dq_opt[k, :]).T
            if k < self.params.N:
                ddpee[[k], :] = self.symkin.eval_ab(
                    q_opt[k, :], dq_opt[k, :], self._u_opt[k, :]).T

        # End-effector position and orientation
        if plot_ee_pose:
            _, ax_c = plt.subplots(2, 1)
            for k in range(3):
                ax_c[0].plot(self._t_opt, pee[:, k] - pee[0, k])
                ax_c[1].plot(self._t_opt, e_rot[:, k])
            ax_c[0].set_ylabel(r"$\Delta p_{ee}$ (m)")
            ax_c[1].set_ylabel(r"$e_{rot}$ (rad)")
            ax_c[0].grid()
            ax_c[1].grid()
            plt.tight_layout()

        # Beam motion
        nq_beam = self.nq - self.nq_arm
        if self.nq > self.nq_arm and plot_pendulum_states:
            _, ax_b = plt.subplots(2, 1)
            for k in range(self.nq_arm, self.nq):
                ax_b[0].plot(self._t_opt, self._x_opt[:, k])
                ax_b[0].axhline(self._x_opt[0, k], ls='--')
            ax_b[0].set_ylabel(r"$\theta$ (m)")
            ax_b[0].grid()

            for k in range(self.model.nx-nq_beam, self.model.nx):
                ax_b[1].plot(self._t_opt, self._x_opt[:, k])
            ax_b[1].set_ylabel(r"$\dot \theta$ (m/s)")
            ax_b[1].set_xlabel(r"$t$ (sec)")
            ax_b[1].grid()
            plt.tight_layout()

        # Control inputs
        if plot_controls:
            _, ax_u = plt.subplots()
            for k, uk in enumerate(self._u_opt.T):
                ax_u.plot(self._t_opt[:-1], uk, 'o-',
                          markersize=2, label=fr"$u_{str(k)}$")
            ax_u.legend(ncol=2)
            ax_u.grid()
            plt.tight_layout()

        # visualize joint positions and velocities
        if plot_joint_states:
            _, ax_j = plt.subplots(2, 1)
            for k in range(self.nq_arm):
                ax_j[0].plot(self._t_opt, self._x_opt[:, k] -
                             self._x_opt[0, k], label=fr"$\Delta q_ {str(k+1)}$")
                ax_j[1].plot(self._t_opt, self._x_opt[:, self.nq+k],
                             label=fr"$\dot q_{str(k)}$")
            ax_j[1].set_xlabel(r"$t$ (sec)")
            ax_j[0].set_ylabel(r"$\Delta q$ (rad)")
            ax_j[1].set_ylabel(r"$\dot q$ (rad/s)")
            ax_j[0].legend(ncol=2)
            ax_j[1].legend(ncol=2)
            ax_j[0].grid()
            ax_j[1].grid()
            plt.tight_layout()

        # visualize cartesian space positions and velocities
        if plot_ee_velocity:
            lbls = [r'$v_x$', r'$v_y$', r'$v_z$',
                    r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
            _, ax_t = plt.subplots(3, 2)
            for k, (ax, dpk) in enumerate(zip(ax_t.T.reshape(-1), dpee.T)):
                ax.plot(self._t_opt, dpk, label=lbls[k])
                ax.set_ylabel(lbls[k])
                ax.grid(alpha=0.5)
            plt.tight_layout()

        if plot_ee_acceleration:
            lbls = [r'$a_x$', r'$a_y$', r'$a_z$', r'$\dot \omega_x$',
                    r'$\dot \omega_y$', r'$\dot \omega_z$']
            _, ax_t = plt.subplots(3, 2)
            for k, (ax, dpk) in enumerate(zip(ax_t.T.reshape(-1), ddpee.T)):
                ax.plot(self._t_opt, dpk, label=lbls[k])
                ax.set_ylabel(lbls[k])
                ax.grid(alpha=0.5)
            plt.tight_layout()
            plt.show()


def poly5_initialization(q_t0, p_tf, R_tf, tf, ts, symkin):
    """ Uses quintic polynomial trajectory planner to find an
    initial guess for the OCP

    :parameter q_t0: intial configuration
    :parameter p_tf: {b} frame position at final time
    :parameter R_tf: {b} frame orientation at final time
    :parameter tf: travel time
    :parameter ts: sampling time
    :parameter symkin: an instance of the symbolic kinematic class
    """
    p_t0 = np.array(symkin.eval_pb(q_t0))
    R_t0 = np.array(symkin.eval_Rb(q_t0))
    H_t0 = RpToTrans(R_t0, p_t0)
    H_tf = RpToTrans(R_tf, p_tf)

    # Design polynomial trajectory
    poly5 = Poly5Trajectory(H_t0, H_tf, tf, ts)
    v = poly5.velocity_reference
    a = poly5.acceleration_reference

    ns = v.shape[0]
    ddq = np.zeros((ns, 7))
    dq = np.zeros((ns, 7))
    q = np.zeros((ns, 7))
    q[0, :] = q_t0.flatten()
    for k, (qk, vk, ak) in enumerate(zip(q, v, a)):
        Jk = np.array(symkin.eval_Jb(qk))
        Jk_pinv = np.linalg.pinv(Jk)
        dq[k, :] = Jk_pinv @ vk

        dJk = np.array(symkin.eval_dJb(qk, dq[k, :]))
        ddq[k, :] = Jk_pinv @ (ak - dJk @ dq[k, :])
        if k < ns-1:
            q[k+1, :] = q[k, :] + ts*dq[k, :]

    return q, dq, ddq


def rotation_error_fcn(fkrot, rotf):
    """ Orientation error wrt initial configuration 
    based on Ajenadros code (Siciliano's book Chapter 3.7)

    :parameter fkrot: a casadi function for computing rotation matrix
    :parameter rotf: final configuration matrix

    :return rot_err: a casadi function for computing rotation error
    """
    q = cs.MX.sym('q', 7)
    Ree = fkrot(q)

    ee_rot_n = cs.Function('rot_n', [q], [Ree[:, 0]])
    ee_rot_s = cs.Function('rot_s', [q], [Ree[:, 1]])
    ee_rot_a = cs.Function('rot_a', [q], [Ree[:, 2]])

    rotf_n = rotf[:, 0]
    rotf_s = rotf[:, 1]
    rotf_a = rotf[:, 2]

    # Axis and angle notation
    rot_err = cs.Function('rot_err', [q],
                          [0.5*(cs.cross(rotf_n, ee_rot_n(q)) + cs.cross(rotf_s, ee_rot_s(q)) +
                                cs.cross(rotf_a, ee_rot_a(q)))])
    return rot_err


def design_optimal_trajectory(task_name: str, N: int=55, supress_vibrations: bool=True, 
                          beam_params='estimated', visualize=False, save_traj=False):
    """ Optimal control of the robot in joint space

    :parameter task_name: name of the task
    :parameter N: the horizon (specifies time T = N*0.01)
    :parameter supress_vibrations: a flag indicating if vibration shoould be suppressed or ignored
    :parameter arm_model: model of the arm could be ['kinematic', 'first order']
    :parameter beam_params: specifies beam parameters ['nominal', 'lower bound', 'upper bound', 'mean', 'random']
    :parameter motion: specifies a motion ['Z', 'Xup', 'Xdown']
    :parameter visualize: specifies whether to visualize the solution
    :parameter save_traj: specifies if the trajectory should be saved or not

    :return: a dictionary with a solution and its resmapled version, and an another dictionary with
            miscelleneous data used in problem formulation
    """
    # Symbolic kinematic model of the setup
    symkin = SymbolicSetupKinematics()

    # Task specification
    task = ICRA_tasks(task_name)

    # Defining ingredients of the OCP
    # Dynamics
    if supress_vibrations:
        model = setup_kinematic_model(symkin)
    else:
        model = arm_kinematic_model()

    # OCP parameters
    ts = 0.01
    w_q, w_dq = 0.01, 1.
    nq = model.nx//2
    if nq == symkin.nq_arm:
        Q = np.diag([w_q]*symkin.nq_arm + [w_dq]*symkin.nq_arm)
    else:
        Q = np.diag([w_q]*symkin.nq_arm + [1.]*symkin.nq_beam +
                    [w_dq]*symkin.nq_arm + [3.]*symkin.nq_beam)

    ocp_params = OCPParameters(ts=ts, N=N, rho=20, Q=Q,
                               suppress_vibrations=supress_vibrations)

    # Boundary constrants
    boundary_constr = {"pb_tf": task.pb_tf, "Rb_tf": task.Rb_tf}
    if supress_vibrations:
        # Dynamic model parametrs
        p = get_beam_params(type=beam_params)

        # Initial state
        theta_t0 = beam_rest_position(task.Rb_t0, p)
        dtheta_t0 = np.zeros_like(theta_t0)
        x_t0 = np.vstack((task.q_t0, theta_t0, task.dq_t0, dtheta_t0))

        # Final pendulum rest position
        theta_tf = beam_rest_position(task.Rb_tf, p)
        boundary_constr["theta_tf"] = theta_tf
        print(f'theta_t0 = {theta_t0}')
        print(f'theta_tf = {theta_tf}')
    else:
        # Dynamic model parametrs
        p = np.zeros((0,1))

        # Initial state
        x_t0 = np.vstack((task.q_t0, task.dq_t0))
    boundary_constr["x_t0"] = x_t0

    # Compute initial guess
    q0, dq0, ddq0 = poly5_initialization(task.q_t0, task.pb_tf, task.Rb_tf, 
                                         N*ts, ts, symkin)
    if supress_vibrations:
        x0 = np.hstack((q0[:N+1, :], theta_t0*np.ones((N+1, 1)),
                       dq0[:N+1, :], np.zeros(((N+1, 1)))))
    else:
        x0 = np.hstack((q0[:N+1, :], dq0[:N+1, :]))
    u0 = ddq0[:N, :]

    # Define and solve optimal control problem
    js_ocp = OptimalControlProblem(model, ocp_params, boundary_constr, symkin)
    t_opt, x_opt, u_opt = js_ocp.solve(p, x0.T, u0.T)
    # t_opt, x_opt, u_opt, slacks = js_ocp.solve()
    print(f"tf = {t_opt[-1]:.4f}, ts = {t_opt[1]-t_opt[0]}")

    # Resample solution
    ts = 0.001
    t_resampled, x_opt_resmapled = js_ocp.resample_solution(ts)
    q_opt_resampled = x_opt_resmapled[:, :7]
    dq_opt_resampled = x_opt_resmapled[:, nq:nq+7]

    # visualize control inpuits and states
    if visualize:
        js_ocp.visualize_solution()

    if save_traj:
        js_ocp.save_solution()

    sol = {"t": t_opt, "x": x_opt, "u": u_opt, "t_res": t_resampled,
           "q_res": q_opt_resampled, "dq_res": dq_opt_resampled, "x_res": x_opt_resmapled}
    misc = {"symkin": symkin, 'model': model}
    return sol, misc
