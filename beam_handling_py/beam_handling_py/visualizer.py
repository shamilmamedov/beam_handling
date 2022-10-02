#!/usr/bin/env python3

import os
import time
import subprocess

class SetupVisualizer:
    def __init__(self) -> None:
        fname = 'js_opt_traj_4rviz.csv'
        if not os.path.isfile(fname):
            raise RuntimeError('File with the trajectry is not found!')

    def play(self, T):
        """
        :parameter T: visualization time
        """
        command = 'roslaunch bh_setup display_setup_pend_fixed.launch'
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        time.sleep(T)
        proc.terminate()
