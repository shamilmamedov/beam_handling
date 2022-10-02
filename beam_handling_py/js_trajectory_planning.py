#!/usr/bin/env python3

import os

from beam_handling_py.input_shaping import (shape_jsreference_traj_js, 
                                            shape_jsreference_traj_cs)
from beam_handling_py.optimal_control import design_optimal_trajectory
from beam_handling_py.visualizer import SetupVisualizer


if __name__ == "__main__":
    # remove saved trajectories
    try:
        os.remove('js_opt_traj_4rviz.csv')
        os.remove('js_opt_traj.csv')
    except FileNotFoundError:
        pass

    task_2_minN = {'T1': 44, 'T2': 46, 'T3': 81}

    task_name = 'T3'
    suppress = False
    N = task_2_minN[task_name]

    sol, misc = design_optimal_trajectory(task_name, N=N, supress_vibrations=suppress,
                                          beam_params='estimated', visualize=False, save_traj=True)

    v = SetupVisualizer().play(20)
