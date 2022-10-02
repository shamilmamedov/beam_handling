#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from beam_handling_py.panda import Limits
from beam_handling_py.data_processing import PandaData1khz


def latexify(fig_width=None, fig_height=None):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.
    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    if fig_width is None:
        fig_width = 3.5  # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean  # height in inches

    params = {'backend': 'ps',
              'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 10,
              'axes.linewidth': 0.5,
              'grid.linewidth': 0.5,

              'legend.fontsize': 10,  # was 10

              'xtick.labelsize': 8,
              'ytick.labelsize': 8,

              'text.usetex': True,
              'text.latex.preamble': r"\usepackage{amsmath} \usepackage{amssymb}",

              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif',
              'font.serif': 'Times',
              'font.size': 8
              }

    matplotlib.rcParams.update(params)


def plot_joint_positions(T: list, Q: list, remove_offset: bool = True):
    """ Plots joint positions of the Panda

    :parameter t: a list of time sequences
    :parameter q: a list of [nx7] matrices for joint positions
    :parameter remove_offset: a flag for removing an offset of the joint pos
    """
    lstyles = ['-', '--', '-.', ':']
    clrs = plt.rcParams['axes.prop_cycle'].by_key()['color'][:7]
    y_lbl = r"$q$ (rad)" if not remove_offset else r"$\Delta q$ (rad)"

    fig, ax = plt.subplots()
    for j, (t, q) in enumerate(zip(T, Q)):
        if remove_offset:
            q -= q[0, :]
        for k, qk in enumerate(q.T):
            ax.plot(t, qk, color=clrs[k],
                    ls=lstyles[j], label=fr"$q{j+1}_{k+1}$")
    ax.legend(ncol=len(Q))
    ax.grid(alpha=0.4)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(y_lbl)
    plt.tight_layout()
    plt.show()


def plot_joint_velocities(T: list, dQ: list, show_limits: bool = False):
    """ Plots joint positions of the Panda

    :parameter T: a list of time sequences
    :parameter dQ: a list of [nx7] matrix of joint positions
    """
    lstyles = ['-', '--', '-.', ':']
    clrs = plt.rcParams['axes.prop_cycle'].by_key()['color'][:7]
    y_lbl = r"$\dot q$ (rad)"

    if show_limits:
        limits = Limits()

    fig, ax = plt.subplots()
    for j, (t, dq) in enumerate(zip(T, dQ)):
        for k, dqk in enumerate(dq.T):
            ax.plot(t, dqk, color=clrs[k], ls=lstyles[j],
                    label=fr"$\dot q{j+1}_{k+1}$")
    if show_limits:
        for k, dqk_max in enumerate(limits.dq_max):
            ax.axhline(dqk_max, color=clrs[k], ls='--')
            ax.axhline(-dqk_max, color=clrs[k], ls='--')
    ax.legend(ncol=len(dQ))
    ax.grid(alpha=0.4)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(y_lbl)
    plt.tight_layout()
    plt.show()


def plot_joint_accelerations(T: list, ddQ: list, show_limits: bool = False):
    """ Plots joint positions of the Panda

    :parameter T: a list of time sequences
    :parameter ddQ: a list of [nx7] matrix of joint positions
    """
    lstyles = ['-', '--', '-.', ':']
    clrs = plt.rcParams['axes.prop_cycle'].by_key()['color'][:7]
    y_lbl = r"$\ddot q$ (rad)"

    if show_limits:
        limits = Limits()

    fig, ax = plt.subplots()
    for j, (t, dq) in enumerate(zip(T, ddQ)):
        for k, dqk in enumerate(dq.T):
            ax.plot(t, dqk, color=clrs[k], ls=lstyles[j],
                    label=fr"$\ddot q{j+1}_{k+1}$")
    if show_limits:
        for k, ddqk_max in enumerate(limits.ddq_max):
            ax.axhline(ddqk_max, color=clrs[k], ls='--')
            ax.axhline(-ddqk_max, color=clrs[k], ls='--')
    ax.legend(ncol=len(ddQ))
    ax.grid(alpha=0.4)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(y_lbl)
    plt.tight_layout()
    plt.show()


def plot_external_wrench(data: PandaData1khz, xlim=None):
    """ Plots external wrenches for a PandaData*

    :parameter data: a member of  PandaData1khz containing log data
    :parameter xlim: limits x axis in the plots
    """
    try:
        ylabels = [r"$F_x$", r"$F_y$", r"$F_z$",
                   r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"]

        _, axs = plt.subplots(3, 2)
        for k, ax in enumerate(axs.T.reshape(-1)):
            ax.plot(data.t, data.wrench_f0[:, k])
            ax.set_ylabel(ylabels[k])
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.grid(which='minor', linestyle=':')
        plt.tight_layout()
        plt.show()
    except AttributeError:
        print("Data doesn't contain external wrench attribute!")


def compare_ee_wrenches(data: list, labels: list, xlim=None, components=None, tif=None, save=False):
    """ Plots the end-effector wrenches for different experiments

    :parameter data: a list of PandaData1khz objects
    :parameter labels: labels of the data in the data list
    :parameter xlim: limits of the x axis
    :parameter components: which components of the wrench to plot
    :parameter tif: initial and final time of the active phase of the trajectory
    :parameter save: a flag for saving the plot
    """
    if components is None:
        components = [0, 1, 2, 3, 4, 5]

    ylabels = np.array([r"$F_x$ (N)", r"$F_y$ (N)", r"$F_z$ (N)",
                        r"$\tau_x$ (N$\cdot$m)", r"$\tau_y$ (N$\cdot$m)", r"$\tau_z$ (N$\cdot$m)"])

    ylabels = ylabels[components]
    n_comp = len(components)
    nc = 2 if n_comp > 3 else 1
    nr = int(np.ceil(n_comp/nc))

    sampling_rate = int(1/(data[0].t[1] - data[0].t[0]))
    if xlim is not None:
        idxs = np.arange(int(xlim[0]*sampling_rate),
                         int(xlim[1]*sampling_rate))
    else:
        idxs = np.arange(0, data[0].wrench_f0[:, 0].shape[0])

    fig, axs = plt.subplots(nr, nc, sharex=True,
                            squeeze=False, figsize=(3.5, 2))
    for d, l in zip(data, labels):
        for k, ax in enumerate(axs.T.reshape(-1)):
            ax.plot(d.t[idxs], d.wrench_f0[idxs, components[k]], label=l)
            if tif:
                ax.axvline(tif[0], color='k', linestyle='--')
                ax.axvline(tif[1], color='k', linestyle='--')
            ax.set_ylabel(ylabels[k])
            ax.grid(alpha=0.5)
    ax.legend(ncol=1, prop={'family': 'monospace'})
    ax.set_xlabel(r'$t$ (s)')
    plt.tight_layout()
    plt.show()


def compare_ee_positions(data: list, labels: list, xlim=None, tif=None):
    """ Plots ee positions for several data with the purpose of compaing them

    :parameter data: a list of PandaData objects
    :parameter labels: list of labels
    :parameter xlim: limits of the x-axis
    :parameter tif: a list of two number that indicate beginning and end of motion
    """
    ylabels = [r"ee$_x$ (m)", r"ee$_y$ (m)", r"ee$_z$ (m)"]

    _, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
    for d, l in zip(data, labels):
        for k, ax in enumerate(axs.T.reshape(-1)):
            # ax.plot(d.t, d.O_T_EE[:,12+k], label=l)
            ax.plot(d.t, d.b_pos[:, k], label=l)
            ax.set_ylabel(ylabels[k])
            if xlim is not None:
                ax.set_xlim(xlim)
            if tif:
                ax.axvline(tif[0], color='k', linestyle='--')
                ax.axvline(tif[1], color='k', linestyle='--')
            ax.grid(which='minor', linestyle=':')
            ax.legend()
            ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


def compare_ee_twists(data: list, labels: list, components=None, xlim=None, tif=None):
    """ Plots ee twist for several data with the purpose of compaing them

    :parameter data: a list of PandaData objects
    :parameter labels: list of labels
    :parameter components: which components of the wrench to plot
    :parameter xlim: limits of the x axis
    :parameter tif: initial and final time of the active phase of the trajectory
    """
    if components is None:
        components = [0, 1, 2, 3, 4, 5]
    ylabels = np.array([r"$v_x$ (m/s)", r"$v_y$ (m/s)", r"$v_z$ (m/s)",
                        r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"])
    ylabels = ylabels[components]
    n_comp = len(components)
    nc = 2 if n_comp > 3 else 1
    nr = int(np.ceil(n_comp/nc))

    _, axs = plt.subplots(nr, nc, figsize=(12, 7))
    # if nr=nc=1 then make axs np array so that later we don't get
    # error while transposing it
    if nr == 1 and nc == 1:
        axs = np.array([axs])
    for d, l in zip(data, labels):
        for k, ax in enumerate(axs.T.reshape(-1)):
            ax.plot(d.t, d.b_twist[:, components[k]], label=l)
            ax.set_ylabel(ylabels[k])
            if xlim is not None:
                ax.set_xlim(xlim)
            if tif:
                ax.axvline(tif[0], color='k', linestyle='--')
                ax.axvline(tif[1], color='k', linestyle='--')
            ax.grid()
            ax.legend()
    plt.tight_layout()
    plt.show()


def compare_joint_velocities(data: list, labels: list, xlim=None):
    """ Compares joint velocities of several experiments

    :parameter data: a list of PandaData objects
    :parameter labels: list of labels
    :parameter xlim: limits of the x axis
    """
    ylabels = [fr"$\dot q_{k+1}$" for k in range(7)]

    sampling_rate = int(1/(data[0].t[1] - data[0].t[0]))
    if xlim is not None:
        idxs = np.arange(int(xlim[0]*sampling_rate),
                         int(xlim[1]*sampling_rate))
    else:
        idxs = np.arange(0, data[0].dq.shape[0])

    _, axs = plt.subplots(3, 2, figsize=(10, 6), sharex=True)
    for d, l in zip(data, labels):
        for k, ax in enumerate(axs.T.reshape(-1)):
            ax.plot(d.t[idxs], d.dq[idxs, k], label=l)
            try:
                ax.plot(d.t[idxs], d.dq_d[idxs, k], '--')
            except AttributeError:
                pass
            ax.set_ylabel(ylabels[k])
            ax.grid(which='minor', linestyle=':')
            ax.legend(loc=1)
    plt.xlabel(r'$t$, (sec)')
    plt.tight_layout()
    plt.show()
