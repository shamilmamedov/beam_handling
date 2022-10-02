#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from beam_handling_py.input_shaping import ZV_input_shaper, ZVD_input_shaper
from beam_handling_py.kinematics_utils import rotm2axang, RpToTrans

# Set plotting parameters
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams["xtick.minor.visible"] =  True
plt.rcParams["ytick.minor.visible"] =  True

class Poly5Trajectory:
    """
    Implements quintic polynomial trajectory planning in the 
    operational space
    """
    def __init__(self, yi, yf, tf, ts=0.001) -> None:
        """
        :param yi: initial position, can be joint position or cartesian pose
        :type yi: a numpy matrix
        :param yf: terminal position, -/-
        :type yf: a numpy matrix
        :param tf: travel time
        :param ts: sampling time
        """
        # check if trajectory has to be planned for cartesian pose
        if yi.shape[0] == 4 & yi.shape[1] == 4:
            self.Ri = yi[:3,:3]
            self.pi = yi[:3,3]
            self.Rf = yf[:3,:3]
            self.pf = yf[:3,3]
            rot = self.Ri.T @ self.Rf
            u_i, self.alpha = rotm2axang(rot)
            self.u = self.Ri @ u_i
        else:
            self.pi = yi
            self.pf = yf
        self.ts = ts
        self.tf = tf
        self.t = np.arange(0, tf+ts, ts).reshape(-1,1)

        # get interpolation function and desing trajectory
        self.r, self.dr, self.ddr = self.interp_fcn(self.t, self.tf)
        self.p, self.dp, self.ddp = self.design_traj()
        
    @staticmethod
    def interp_fcn(t, tf):
        """ Interpolation function for quintic polynomial. For 
        more information refer to "Modeling, Identification and Control of Robots"

        :parameter t: [Nx1] time samples
        :parameter tf: travel time
        """
        ttf = t/tf
        r = 10*ttf**3 - 15*ttf**4 + 6*ttf**6
        dr = 1/tf*(30*ttf**2 - 60*ttf**3 + 30*ttf**4)
        ddr = 1/tf**2*(60*ttf - 180*ttf**2 + 120*ttf**3)
        return r, dr, ddr

    def design_traj(self):
        """ Design a trajectory 
        """
        # Translational motion
        delta_p = self.pf - self.pi # amplitude
        p = self.pi.T + self.r*delta_p.T
        dp = self.dr*delta_p.T
        ddp = self.ddr*delta_p.T
        
        # Rotational motion
        try:
            dalpha_t = self.alpha*self.dr
            omega = dalpha_t*self.u.T

            ddalpha_t = self.alpha*self.ddr
            domega = ddalpha_t*self.u.T
            
            dp = np.hstack((dp, omega))
            ddp = np.hstack((ddp, domega))
        except AttributeError:
            pass

        return p, dp, ddp

    def shape_velocity_reference(self, wn, zeta, type="ZV", accel=False):
        """ Shapes velocity reference signal
        
        :parameter wn: natural frequcny of the system
        :parameter zeta: damping ration
        :parameter tpye: type of the shaper
        :parameter accel: a flag inditicating if the trajectory should be accelerated
                            to match the travel tme of the original trajectory
        """
        # get time and amplitude of impulses
        if type == "ZV":
            ti, A = ZV_input_shaper(wn, zeta, self.ts)
        elif type == "ZVD":
            ti, A = ZVD_input_shaper(wn, zeta, self.ts)
        else:
            raise NameError(f"Inpush shaper called {type} doesn't exist")

        idx = [int(x*1000) for x in ti]
        shaper = np.zeros((idx[-1]+1,))
        shaper[idx] = A

        # Perform colvolution of the original rerference with
        # impulses of the input shaper
        if accel:
            # find an accelration constant, calculate new time samples
            # and calculate interpolation function for new time
            k = (self.tf - ti[-1])/self.tf
            t_accel = np.arange(0, k*self.tf+self.ts, self.ts).reshape(-1,1)
            _, self.dr_accel = self.interp_fcn(t_accel, t_accel[-1])

            dr_shaped = np.convolve(self.dr_accel[:,0], shaper).reshape(-1,1)
        else:
            dr_shaped = np.convolve(self.dr[:,0], shaper).reshape(-1,1)
        
                
        delta_p = self.pf - self.pi # amplitude
        dq_shaped = dr_shaped*delta_p.T

        try:
            dalpha_t_shaped = self.alpha*dr_shaped
            omega_shaped = dalpha_t_shaped*self.u.T
            
            dq_shaped = np.hstack((dq_shaped, omega_shaped))
        except AttributeError:
            pass

        self.dp_shaped =  dq_shaped
        self.t_shaped = self.ts*np.arange(0, self.dp_shaped.shape[0]).reshape(-1,1)

    @property
    def velocity_reference(self):
        return self.dp

    @property
    def acceleration_reference(self):
        return self.ddp

    @property
    def shaped_velocity_reference(self):
        return self.dp_shaped


def plot_trajectory(t, y, labels):
    _, ax = plt.subplots()
    for dp, l in zip(y.T, labels):
        ax.plot(t, dp, label=l)
    ax.set_xlabel("t (sec)")
    ax.legend()
    plt.tight_layout()
    plt.show()

