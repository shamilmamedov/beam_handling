#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import detrend
from scipy.integrate import simps
import pandas as pd

from beam_handling_py.data_processing import PandaData1khz
import beam_handling_py.plotting as plotting


# plt.style.use(['science','ieee'])

def plot_tau(data, labels, frame: str = 'base', xlim=None, save_figure: bool = False):
    assert (frame in ['base', 'local'])

    sampling_rate = int(1/(data[0].t[1] - data[0].t[0]))
    if xlim is not None:
        idxs = np.arange(int(xlim[0]*sampling_rate),
                         int(xlim[1]*sampling_rate))
    else:
        idxs = np.arange(0, data[0].wrench_f0[:, 0].shape[0])

    linestyles = ['-', '--', '-.', ':', '-']
    si = []
    fig, ax = plt.subplots(figsize=(3.5, 2))
    for d, l, ls in zip(data, labels, linestyles):
        if frame == 'base':
            t = ax.plot(d.t[idxs], d.wrench_f0[idxs, 4], ls=ls, label=l)
        else:
            t = ax.plot(d.t[idxs], d.wrench_b[idxs, 5], ls=ls, label=l)
        si.append(t[0])
    if frame == 'base':
        ax.set_ylabel(r"$\hat \tau_y^0$ [N$\cdot$m]")
    else:
        ax.set_ylabel(r"$\hat \tau_z^b$ [N$\cdot$m]")
    ax.set_xlim(xlim)
    ax.grid(alpha=0.5)
    ax.set_xlabel(r'$t$ [s]')
    # ax.legend(ncol=1, prop={'family': 'monospace'})
    plt.tight_layout()
    plt.show()

    if save_figure:
        fig.savefig('tauy.svg', format='svg', dpi=600, bbox_inches='tight')


def plot_tau_all_tasks(data, xlim, save_figure: bool = False):
    data_xdown, data_zx, data_3D = data

    sampling_rate = 1000
    idxs = np.arange(int(xlim[0]*sampling_rate),
                     int(xlim[1]*sampling_rate))

    linestyles = ['-', '--', '-.', ':', '-']
    fig, axs = plt.subplots(1, 3)

    for d, ls in zip(data_xdown, linestyles):
        axs[0].plot(d.t[idxs], d.wrench_b[idxs, 5], ls=ls)
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel(r'$\hat \tau_z^b$ [$\mathrm{N}\cdot\mathrm{m}$]')
    axs[0].set_xlabel(r'$t$ [$\mathrm{s}$]')

    for d, ls in zip(data_zx, linestyles):
        axs[1].plot(d.t[idxs], d.wrench_b[idxs, 5], ls=ls)
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_xlabel(r'$t$ [$\mathrm{s}$]')

    for d, ls in zip(data_3D, linestyles):
        axs[2].plot(d.t[idxs], d.wrench_b[idxs, 5], ls=ls)
    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[2].set_xlabel(r'$t$ [$\mathrm{s}$]')

    [ax.set_xlim(xlim) for ax in axs]
    [ax.grid(alpha=0.5) for ax in axs]

    plt.tight_layout()
    plt.show()

    if save_figure:
        fig.savefig('tau_T123.svg', format='svg', dpi=600, bbox_inches='tight')


def integrate_residual_vibrations(data: PandaData1khz, t_exec: float, signal: str = 'tau',
                                  frame: str = 'base', visualize: bool = False):
    """ Integrates the residual vibrations
    :parameter data: logged data correspnding to a trajectory
    :parameter t_exec: trajectory execution time (active phase of trajectory)
    """
    assert (frame in ['base', 'local'])

    signal2idx = {'tau': 4, 'F': 2} if frame == 'False' else {'tau': 5, 'F': 1}
    idx = signal2idx[signal]

    # Define beginning and the end of the signal to be integrated
    # in the beginning there is a padding of 0.1 second, time at the end
    # in my choice, perhaps can be removed
    t0 = 0.1 + t_exec + 0.1
    t_intg = 5
    tf = t0 + t_intg

    # Get the signal and time
    if frame == 'base':
        y = data.wrench_f0[int(t0/data.ts):, idx]
    else:
        y = data.wrench_b[int(t0/data.ts):, idx]
    y_detrend = detrend(y, type='constant')
    t = data.t[int(t0/data.ts):]

    y_2intg = y_detrend[:int(t_intg/data.ts)]
    t_2intg = t[:int(t_intg/data.ts)].flatten()

    # Integrate the signal
    vibr_intg = simps(abs(y_2intg), t_2intg)

    if visualize:
        _, ax = plt.subplots()
        if frame == 'base':
            ax.plot(data.t, data.wrench_f0[:, idx])
        else:
            ax.plot(data.t, data.wrench_b[:, idx])
        ax.plot(t, y_detrend)
        ax.plot(t_2intg, abs(y_2intg))
        ax.axhline(0., ls='--', color='black')
        ax.axvline(tf, ls='--', color='black')
        ax.axvline(t0, ls='--', color='black')
        plt.show()

    return vibr_intg


def analyze_OCP_vs_ZV_shapers(task_names, frame: str = 'local', save_figure: bool = False):
    """ In these experiments the end-effector angular velocity
    is not fixed.
    """
    # Define path to folders with data
    n = len(task_names)
    df = 'log/vibration_suppression_ICRA/'

    if 'T1' in task_names:
        ocp_Xdown_44 = PandaData1khz(df + 'aOCP_T1_0.44s.csv')
        voscp_Xdown_44 = PandaData1khz(df + 'OCP_T1_0.44s.csv')
        voscp_Xdown_50 = PandaData1khz(df + 'OCP_T1_0.50s.csv')
        voscp_Xdown_56 = PandaData1khz(df + 'OCP_T1_0.56s.csv')
        csZV_Xdown_56 = PandaData1khz(df + 'csZVIS_T1_0.56s.csv')

        data_xdown = [ocp_Xdown_44, voscp_Xdown_44,
                      voscp_Xdown_50, voscp_Xdown_56, csZV_Xdown_56]
        labels = [f"{'w/o':<4s}{'0.44s':<4s}", f"{'w/':<4s}{'0.44s':<4s}",
                  f"{'w/':<4s}{'0.50s':<4s}", f"{'w/':<4s}{'0.56s':<4s}", f"{'ZV':<4s}{'0.56s':<4s}"]
        t_execs = [0.44, 0.44, 0.50, 0.56, 0.56]

        if n == 1:
            plotting.latexify()
            plot_tau(data_xdown, labels, frame, xlim=[0, 2], save_figure=save_figure)
            data = data_xdown

    if 'T2' in task_names:
        ocp_ZX_46 = PandaData1khz(df + 'aOCP_T2_0.46s.csv')
        vsocp_ZX_46 = PandaData1khz(df + 'OCP_T2_0.46s.csv')
        vsocp_ZX_55 = PandaData1khz(df + 'OCP_T2_0.55s.csv')
        vsocp_ZX_62 = PandaData1khz(df + 'OCP_T2_0.62s.csv')
        csZV_ZX_62 = PandaData1khz(df + 'csZVIS_T2_0.62s.csv')

        data_zx = [ocp_ZX_46, vsocp_ZX_46,
                   vsocp_ZX_55, vsocp_ZX_62, csZV_ZX_62]
        labels = ['w/o  0.46s', 'w/ 0.46s', 'w/ 0.55s', 'w/ 0.62s', 'ZV 0.62s']
        t_execs = [0.46, 0.46, 0.55, 0.62, 0.62]

        if n == 1:
            plotting.latexify()
            plot_tau(data_zx, labels, frame, xlim=[0, 2], save_figure=save_figure)
            data = data_zx

    if 'T3' in task_names:
        ocp_81 = PandaData1khz(df + 'aOCP_T3_0.81s.csv')
        vsocp_81 = PandaData1khz(df + 'OCP_T3_0.81s.csv')
        vsocp_90 = PandaData1khz(df + 'OCP_T3_0.90s.csv')
        vsocp_99 = PandaData1khz(df + 'OCP_T3_0.99s.csv')
        jsZV_99 = PandaData1khz(df + 'jsZVIS_T3_0.99s.csv')

        data_3D = [ocp_81, vsocp_81, vsocp_90, vsocp_99, jsZV_99]
        labels = ['w/o  0.81s', 'w/ 0.81s',
                  'w/ 0.90s', 'w/ 0.99s', 'jsZV 0.99s']
        t_execs = [0.81, 0.81, 0.90, 0.99, 0.99]

        if n == 1:
            plotting.latexify()
            plot_tau(data_3D, labels, frame, xlim=[0, 2.5], save_figure=save_figure)
            data = data_3D

    if n == 3:
        xlim = [0, 2]
        plotting.latexify(fig_width=7.16, fig_height=2)
        plot_tau_all_tasks([data_xdown, data_zx, data_3D], xlim, save_figure=save_figure)

    if n == 1:
        vibr_ingts = np.array([integrate_residual_vibrations(d, t_exec, frame=frame)
                            for d, t_exec in zip(data, t_execs)])
        vibr_ingts = vibr_ingts * 100 / vibr_ingts[0]
        print(pd.DataFrame({'traj': labels, 'vibr_intg': vibr_ingts}))

    

if __name__ == "__main__":
    # analyze_OCP_vs_ZV_shapers_constr_orient()
    analyze_OCP_vs_ZV_shapers(['T1'], save_figure=False)