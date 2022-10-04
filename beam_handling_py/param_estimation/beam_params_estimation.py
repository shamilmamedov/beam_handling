#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from scipy.signal import find_peaks
from scipy import signal

import beam_handling_py.data_processing as dp
import beam_handling_py.plotting as plotting


def estimate_sos_parameters(t, y, verbose=False, visualize=False):
    """
    Estimating parameters of the second order system:
    natural frequency and damping ratio using peaks of the signal

    :param t: time sequnce
    :param y: output signal (residual vibrations)
    :param verbose: a flag for printing additional info
    :param visualize: a flag for visualizing period of oscillations
                      and log decrement as a function of # peaks 

    :return: natural frequency wn, damping ratio zeta and a dictionaty 
            containing peaks, Ti and delta_i
    """
    # Find peaks of the oscillations
    dt = t[1] - t[0]
    freq = int(1/dt)

    if freq == 120:  # motion capture case
        peaks, _ = find_peaks(y, distance=10, height=0)
    else:  # pandas measurements
        dist = 120 if dt <= 0.01 else 10
        peaks, _ = find_peaks(y, distance=dist, height=0)

    # Estimate period of damped oscillations
    Ti = (t[peaks[1:]] - t[peaks[0]])/np.arange(1, len(peaks))
    T_mean = np.mean(Ti)
    T_median = np.median(Ti)

    # Estimate logarithmic decrement
    delta_i = np.log(y[peaks[0]]/y[peaks[1:]])/np.arange(1, len(peaks))
    delta_mean = np.mean(delta_i)
    delta_median = np.median(delta_i)
    if verbose:
        print(f"Period estimates for different n\n {Ti}")
        print(f"Log decrement estimates for different n\n {delta_i}")

    # Visualize Ti and delta_i
    if visualize:
        _, ax = plt.subplots(2, 1, sharex=True)
        ax[0].scatter(np.arange(Ti.size)+1, Ti)
        ax[0].axhline(T_mean, linestyle='dashed', color='k',
                      label=f'mean = {T_mean:.3f}')
        ax[0].axhline(T_median, linestyle='dashed', color='r',
                      label=f'median = {T_median:.3f}')
        ax[0].set_ylabel(r'$T_i$ (s)')
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[0].legend()

        ax[1].scatter(np.arange(delta_i.size)+1, delta_i)
        ax[1].axhline(delta_mean, linestyle='dashed', color='k',
                      label=f'mean = {delta_mean:.3f}')
        ax[1].axhline(delta_median, linestyle='dashed', color='r',
                      label=f'median = {delta_median:.3f}')
        ax[1].set_ylabel(r'$\delta_i$')
        ax[1].set_xlabel(r'$t$ (s)')
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    # Choose between mean and median
    T = T_median
    delta = delta_median

    # Compute damping ratio
    zeta = delta/np.sqrt(4*np.pi**2 + delta**2)

    # Compute natural frequency
    wn = np.sqrt(4*np.pi**2 + delta**2)/T

    return wn, zeta, {'peaks': peaks, 'T_i': Ti, 'delta_i': delta_i}


def estimate_beam_parameters(dd, filter_output=False, est_interval=None, visualize_fit=True):
    """
    Estimating beam parameters -- natural frequency and damping ratio 
    -- from fast point-to-point motion. The function takes several realization
    (repetitions) of the same trajectory, averages it, filters with zero-phase 
    filter and then estimation parameters;

    :param dd: a dictionary with dataset description
    :param visualize fit: a flag indicating if to visualize fit
    """
    # Load data
    data, _ = load_param_estimation_data(dd)

    # Average wrench and time measurements
    idx = dd['tau_idx']
    # idx = 5 # 3 for Z, 5 for Xdown
    t_avrg, w_avrg = average_wrench(data, [idx])

    F_avrg = w_avrg[0]

    # Filter force measurement
    if filter_output:
        filt_freq = 10
        F_avrg_filt = filter_zero_phase(F_avrg, f_filt=filt_freq)
    else:
        F_avrg_filt = F_avrg

    # Choosing interval for parameter estimation
    ts = t_avrg[1] - t_avrg[0]
    if est_interval == None:
        ti = 3.5 if dd['tf'] is np.nan else float(dd['tf'][:-1]) + 2
        tf = ti + 6
    else:
        ti, tf = est_interval[0], est_interval[1]
    idx_range = np.arange(int(ti/ts), int(tf/ts))  # !!!!!!!!
    t_train = t_avrg[idx_range]
    F_train = F_avrg_filt[idx_range]
    F_train_mean = np.mean(F_train)

    # Estimate second order system parmaeters
    wn, zeta, buf = estimate_sos_parameters(t_train, F_train-F_train_mean,
                                            verbose=True, visualize=visualize_fit)
    print(f'Estimated beam parameters: wn={wn:.5f}, zeta={zeta:.5f}')
    peaks = buf['peaks']

    # Visualize fit
    if visualize_fit:
        # Predict output
        wd = wn*np.sqrt(1 - zeta**2)
        F0 = F_train[peaks[0]] - F_train_mean
        tt = t_train[peaks[0]:] - t_train[peaks[0]]
        F_pred = np.exp(-zeta*wn*tt)*(F0*np.cos(wd*tt) +
                                      zeta*wn*F0*np.sin(wd*tt)/wd) + F_train_mean

        _, ax = plt.subplots()
        ax.plot(t_train, F_avrg[idx_range], label='real')
        # ax[0].plot(t_train, F_train, label='real filt')
        ax.axhline(F_train_mean, color='k')
        ax.scatter(t_train[peaks], F_train[peaks])
        ax.plot(t_train[peaks[0]:], F_pred, label='model')
        ax.plot(t_train[peaks[0]:], F_train_mean +
                F_train[peaks[0]:] - F_pred, label='error')
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$F$ (N)")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Compute relative residual error
        rre_F = 100. * \
            np.linalg.norm(F_train[peaks[0]:] - F_pred) / \
            np.linalg.norm(F_train[peaks[0]:])
        print(f"rre F train = {rre_F}%")

    return wn, zeta


def get_param_estimation_data_description():
    """ Returns parameter estimation dataset description as pandas dataframe

    :parameter beam: specifies the beam used in experiments, can take values ['old', 'new']
    :parameter dataset: specifies which dataset to use; valid only for the 'new' beam

    :return: description of all experiments as pandas dataframe
    """
    axes = [*['Z']*3, *['Xup']*3, *['Xdown']*3]
    amplts = ['15cm', '25cm', '35cm']*3
    tfs = ['0.42s', '0.50s', '0.61s', *['0.35s', '0.44s', '0.57s']*2]
    n_expers = [*[6]*3]*3
    F_idxs = [*[2]*3, *[0]*6]
    tau_idxs = [4]*9

    df = pd.DataFrame({'axis': axes, 'amplt': amplts, 'tf': tfs,
                       'n_exper': n_expers, 'F_idx': F_idxs, 'tau_idx': tau_idxs})
    return df


def load_param_estimation_data(dataset_description, resample=True):
    """" loads data for parameter estimation by creating PandData instance
    for each experiement

    :param dataset_description: description of the dataset as a dict
    :param resample: a flag indicating if one needs to resample raw data
    :param n_impulse: indicates which impulse to use; For impulse response 
                averaging doesn't make sense

    :return: a list of PandaData objects and a list of labels for each data
    """
    # Parse dataset description dictionary
    axis = dataset_description['axis']
    amplt = dataset_description['amplt']
    tf = dataset_description['tf']
    n_exp = dataset_description['n_exper']

    # Construc path to dataset files
    log_dir = 'data/param_estimation_ICRA'

    logs = [f'ocp_{axis}_{amplt}_{tf}_{k+1}.csv' for k in range(n_exp)]
    path_to_logs = [os.path.join(log_dir, log) for log in logs]

    # Create a list of PandaData objects
    data = [dp.PandaData1khz(p, resample=resample) for p in path_to_logs]
    data_labels = [f'{axis}_{amplt}_{k+1}' for k in range(n_exp)]
    return data, data_labels


def print_dataset_info(d):
    """ Prints dataset information
    """
    print(f'axis: {d["axis"]}, amplitude: {d["amplt"]}, ' +
          f'tf: {d["tf"]}, n_experiments: {d["n_exper"]}')


def average_wrench(data, idxs):
    """ Averages several realizations (implementations) of the same motion.
    The function is developed mainly for wrench and twist measuremetns

    :param data: a list containing log of several trajectories
    :param idxs: a list indexes of the attribute

    :return: averaged time, force and torque
    """
    # Find min number of samples among all logs

    # Extract necessary data and average it
    n_samples = min([d.wrench_0.shape[0] for d in data])
    F = [np.hstack(tuple(d.wrench_0[:n_samples, [F_idx]]
                    for d in data)) for F_idx in idxs]
    avrg = [np.mean(Fk, axis=1) for Fk in F]

    t = np.hstack(tuple(d.t[:n_samples, :] for d in data))
    return np.mean(t, axis=1), avrg


def filter_zero_phase(F, f_filt=20):
    """ Filters a force or torque measurement using Butterworth filter
    of 4th order

    :param F: an array of force or torque measurement
    :param f_filt: filters cut-off frequency

    :return: filtered torque or force
    """
    # Define filter parameters
    fs = 1e+2  # sampling frequency
    fnyq = fs/2  # nyquist frequency

    # Butterworth filter parameters
    filt_order = 4
    f_filt_norm = f_filt/fnyq  # rnomalized cut-off frequency
    b, a = signal.butter(filt_order, f_filt_norm)

    # Filter data and estiamate velocities and accelerations
    F_filt = signal.filtfilt(b, a, F)
    return F_filt


def analyze_Ti_deltai(save_fig=False):
    """ Analyse natural frequency and  damping ratio for differnt data
    """
    axis2orient = {'Z': r'$\mathcal{O}_1$',
                   'Xdown': r'$\mathcal{O}_3$', 'Xup': r'$\mathcal{O}_2$'}
    amplt2superscr = {'15cm': r'$-1$', '25cm': r'$-2$', '35cm': r'$-3$'}

    # get data description
    dd = get_param_estimation_data_description()

    Ti_list = []
    deltai_list = []
    label_list = []
    label_list_ICRA23 = []
    wn_list = []
    zeta_list = []
    for k in range(9):
        # Get expriment description
        expr_descr = dd.iloc[k]
        label_list.append(expr_descr['axis'] + '-' + expr_descr['amplt'])
        label_list_ICRA23.append(
            axis2orient[expr_descr['axis']] + amplt2superscr[expr_descr['amplt']])

        # Load data
        data, _ = load_param_estimation_data(expr_descr, resample=True)

        # Average wrench and time measurements
        t_avrg, w_avrg = average_wrench(data, [expr_descr['tau_idx']])
        F_avrg = w_avrg[0]

        # Filter force measurement
        # filt_freq = 15
        # F_avrg_filt = filter_zero_phase(F_avrg, f_filt=filt_freq)
        F_avrg_filt = F_avrg

        # Choosing interval for parameter estimation
        ts = t_avrg[1] - t_avrg[0]
        est_interval = dataset2est_interval[k]
        ti, tf = est_interval[0], est_interval[1]
        idx_range = np.arange(int(ti/ts), int(tf/ts))
        t_train = t_avrg[idx_range]
        F_train = F_avrg_filt[idx_range]
        F_train_mean = np.mean(F_train)

        # Estimate second order system parmaeters
        wn, zeta, buf = estimate_sos_parameters(t_train, F_train-F_train_mean,
                                                verbose=False, visualize=False)
        wn_list.append(wn)
        zeta_list.append(zeta)
        Ti_list.append(buf['T_i'])
        deltai_list.append(buf['delta_i'])

    df = pd.DataFrame(
        {"experiment": label_list, "wn": wn_list, "zeta": zeta_list})
    print(df)

    # Set matplotlib params for paper quality figure
    if save_fig:
        plotting.latexify(3.5, 3)

    clrs = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
    clrs = clrs*3
    ms = 20
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(3.5, 3))
    for li, li_ICRA, Ti, c in zip(label_list, label_list_ICRA23, Ti_list, clrs):
        if li.split('-')[0] in ['Xup', r'$\mathcal{O}_2$']:
            axs[0].scatter(np.arange(0, len(Ti)), Ti, s=ms,
                           marker='x', color=c, label=li_ICRA)
        elif li.split('-')[0] in ['Xdown', r'$\mathcal{O}_3$']:
            axs[0].scatter(np.arange(0, len(Ti)), Ti, s=ms,
                           marker='*', color=c, label=li_ICRA)
        else:
            axs[0].scatter(np.arange(0, len(Ti)), Ti, s=ms,
                           marker='o', color=c, label=li_ICRA)
    axs[0].grid(alpha=0.5)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel(r'$T$ [s]')
    axs[0].set_xlim([-1, 15.5])

    si = []
    for li, li_ICRA, deltai, c in zip(label_list, label_list_ICRA23, deltai_list, clrs):
        if li.split('-')[0] == 'Xup':
            t = axs[1].scatter(np.arange(0, len(deltai)),
                               deltai, s=ms, marker='x', color=c, label=li_ICRA)
        elif li.split('-')[0] == 'Xdown':
            t = axs[1].scatter(np.arange(0, len(deltai)),
                               deltai, s=ms, marker='*', color=c, label=li_ICRA)
        else:
            t = axs[1].scatter(np.arange(0, len(deltai)),
                               deltai, s=ms, marker='o', color=c, label=li_ICRA)
        si.append(t)
    axs[1].grid(alpha=0.5)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_xlabel(r'$n$')
    axs[1].set_ylabel(r'$\delta$')
    axs[1].set_xlim([-0.5, 14.5])
    fig.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('Ti_deltai.svg', format='svg', dpi=1200, bbox_inches='tight')


if __name__ == "__main__":
    dd = get_param_estimation_data_description()
    train_dataset = 0

    print("Training dataset description: ")
    print_dataset_info(dd.iloc[train_dataset])

    dataset2est_interval = {0: [4.5, 10], 1: [4.5, 10], 2: [6, 10],
                            3: [6, 10], 4: [5, 10], 5: [3, 10],
                            6: [6, 10], 7: [5, 10], 8: [3, 10]}

    # Estimate parameters for an individual experiments
    # est_interval = dataset2est_interval[train_dataset]
    # wn, zeta = estimate_beam_parameters(dd.iloc[train_dataset], est_interval=est_interval,
    #                                     visualize_fit=True)

    # Analyze parameter estimates for all experiments
    analyze_Ti_deltai(save_fig=False)
