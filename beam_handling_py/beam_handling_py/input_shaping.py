#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import casadi as cs

from  beam_handling_py.kinematics_utils import VelocityLevelIKSolver

def closest_sample_time(x, ts):
    """ Returns closest sample time that complies
    with sampling rate of the system
    """
    # return ts*np.round(x/ts)
    return ts*np.array([int(item/ts) for item in x])

def ZV_input_shaper(wn, zeta, ts):
    """
    Zero Vibration (ZV) input shaper. The algorithm
    for finding time instances of impulses is intended
    for continiuous systems, thereby timeintances do not
    necessarily match sampling time. For that reason
    I round the time instace to closes sampling time.
    """
    # damped natural frequency
    wd = wn*np.sqrt(1 - zeta**2)
    # damped period of vibrations
    Td = 2*np.pi/wd 

    # amplitude and time instances of impulses
    K = np.exp(-zeta*np.pi/np.sqrt(1 - zeta**2))
    A = np.array([1/(1+K), K/(1+K)])
    t_raw = np.array([0, Td/2])
    return closest_sample_time(t_raw, ts), A

def ZVD_input_shaper(wn, zeta, ts):
    """
    Zero Vibration and Derivative shaper (ZVD). The algorithm
    for finding time instances of impulses is intended
    for continiuous systems, thereby timeintances do not
    necessarily match sampling time. For that reason
    I round the time instace to closes sampling time.
    """
    # damped natural frequency
    wd = wn*np.sqrt(1 - zeta**2)
    # damped period of vibrations
    Td = 2*np.pi/wd 

    # amplitude and time instances of impulses
    K = np.exp(-zeta*np.pi/np.sqrt(1 - zeta**2))
    den = 1 + 2*K + K**2
    A = np.array([1/den, 2*K/den, K**2/den])
    t_raw = np.array([0, Td/2, Td])
    return closest_sample_time(t_raw, ts), A

def UM_ZV_input_shaper(wn, zeta, ts):
    """" Unity magnitude negative ZV input shaper 
    """
    A = np.array([1., -1., 1])

    T = 2*np.pi/wn
    M = np.array([[0., 0., 0., 0.],
                [0.16724, 0.27242, 0.20345, 0.],
                [0.33323, 0.00533, 0.17914, 0.20125]])
    zeta_vec = T*np.array([[1., zeta, zeta**2, zeta**3]]).T
    t = M @ zeta_vec
    return closest_sample_time(t.flatten(), ts), A

def UM_ZVD_input_shaper(wn, zeta, ts):
    """ Unity magnitude negative ZVD input shaper
    """
    A = np.array([1, -1, 1, -1, 1])

    T = 2*np.pi/wn
    M = np.array([[0., 0., 0., 0.],
                [0.08945, 0.28411, 0.23013, 0.16401],
                [0.36613, -0.08833, 0.24048, 0.17001],
                [0.64277, 0.29103, 0.23262, 0.43784],
                [0.73228, 0.00992, 0.49385, 0.38633]])
    zeta_vec = T*np.array([[1., zeta, zeta**2, zeta**3]]).T
    t = M @ zeta_vec
    return closest_sample_time(t.flatten(), ts), A

def PS_ZV_input_shaper(wn, zeta, ts):
    """ Partial sum negative ZV input shaper
    """
    A = np.array([1., -2., 2])
    
    T = 2*np.pi/wn
    M = np.array([[0., 0., 0., 0.],
                [0.20970, 0.22441, 0.08028, 0.23124],
                [0.29013, 0.09557, 0.10346, 0.24624]])
    zeta_vec = T*np.array([[1., zeta, zeta**2, zeta**3]]).T
    t = M @ zeta_vec
    return closest_sample_time(t.flatten(), ts), A

def input_shaper(wn, zeta, ts, type='ZV'):
    """ A meta function for input shapers
    """
    if type=='ZV':
        return ZV_input_shaper(wn, zeta, ts)
    elif type=='UM_ZV':
        return UM_ZV_input_shaper(wn, zeta, ts)
    elif type=='PS_ZV':
        return PS_ZV_input_shaper(wn, zeta, ts)
    elif type=='ZVD':
        return ZVD_input_shaper(wn, zeta, ts)
    elif type=='UM_ZVD':
        return UM_ZVD_input_shaper(wn, zeta, ts)
    else:
        raise NotImplementedError


def shape_jsreference_traj_js(dq_ref, wn, zeta, ts, shaper='ZV', save=False):
    """ Shape joint velocity refrence trajectory

    :parameter dq_ref: [ns x 7] matrix of reference joint velocities
    :parameter wn: natural frequency
    :parameter zeta: damping ratio
    :parameter ts: sampling time of the reference trajectory
    :parameter shaper: type of input shaper
    :parameter save: a flag that indicates if the trajecrory should be saved

    :retrun:    t_shaped - time vector of the shaped trajectory
                dq_ref_shaped - shaped reference velocity
    """
    # Design shaper
    ti, A = input_shaper(wn, zeta, ts, type=shaper)

    # Process shaper output
    t_idx = [int(x/ts) for x in ti]
    impulses = np.zeros((t_idx[-1]+1,))
    impulses[t_idx] = A

    # Shape trajectories
    nj = dq_ref.shape[1]
    tmp = tuple(np.convolve(dq_ref[:,k].flatten(), impulses).reshape(-1,1) for k in range(nj))
    dq_ref_shaped = np.hstack(tmp)
    ns_shaped = dq_ref_shaped.shape[0]
    t_shaped = ts*np.arange(0, ns_shaped)
    print(f"tf_{shaper} = {t_shaped[-1]}")

    if save:
        dq_ref_shaped = np.pad(dq_ref_shaped, ((100,100), (0,0)), mode='constant')
        np.savetxt('js_opt_traj_shaped.csv', dq_ref_shaped,  fmt='%.20f', delimiter=',')
    return t_shaped, dq_ref_shaped


def shape_jsreference_traj_cs(q_ref, dq_ref, wn, zeta, ts, shaper='ZV', save=False, debug=False):
    """ Shapes reference trajectory in Cartesian space by first performing
    forward velocity kinematics, shaping and then going back to joint space

    :parameter q_ref: [ns x 7] matrix of reference joint positions
    :parameter dq_ref: [ns x 7] matrix of reference joint velocities
    :parameter wn: natural frequency
    :parameter zeta: damping ratio
    :parameter ts: sampling time of the reference trajectory
    :parameter shaper: type of input shaper
    :parameter save: a flag that indicates if the trajecrory should be saved

    :retrun:    t_shaped - time vector of the shaped trajectory
                dq_ref_shaped - shaped reference velocity
    """
    # Load jacobian function
    eval_vee = cs.Function.load('beam_insertion_py/casadi_fcns/eval_vee.casadi')
    Jbeam_cs = cs.Function.load('beam_insertion_py/casadi_fcns/eval_Jpend.casadi')
    
    # Compute the end-effector (beam-frame) velocity
    ns = q_ref.shape[0]
    v_ref = np.zeros((ns,6))
    for k, (qk, dqk) in enumerate(zip(q_ref, dq_ref)):
        v_ref[k,:] = np.array(eval_vee(qk, dqk)).flatten()
    
    # Input shaping of the Carteisna velocity reference
    ti, A = input_shaper(wn, zeta, ts, type=shaper)

    # process shaper output
    t_idx = [int(x/ts) for x in ti]
    impulses = np.zeros((t_idx[-1]+1,))
    impulses[t_idx] = A

    # Shape end-effector velocity
    tmp = tuple(np.convolve(v_ref[:,k].flatten(), impulses).reshape(-1,1) for k in range(6))
    v_ref_shaped = np.hstack(tmp)
    ns_shaped = v_ref_shaped.shape[0]
    t_shaped = ts*np.arange(0, ns_shaped)

    # Compute simple numerical inverse kinematics
    iksolver = VelocityLevelIKSolver(ignore_velocity_constraints=True)

    dq_shaped = np.zeros((ns_shaped, 7))
    q_shaped = np.zeros((ns_shaped, 7))
    q_shaped[0,:] = q_ref[0,:]
    for k, (qk, vk) in enumerate(zip(q_shaped, v_ref_shaped)):
        J = np.array(Jbeam_cs(qk))
        # dq_shaped[k,:] = np.linalg.pinv(J) @ vk
        dq_shaped[k,:] = iksolver.solveIK(vk, qk).flatten()
        if k < ns_shaped-1:
            q_shaped[k+1,:] = q_shaped[k,:] + ts*dq_shaped[k,:]

    if debug:
        k = ns_shaped//2
        np.testing.assert_allclose(np.array(eval_vee(q_shaped[k,:], dq_shaped[k,:])).flatten(),
                                    v_ref_shaped[k,:])

        t = np.arange(0, ns)*ts
        _, axs = plt.subplots(2,3)
        for ax, vk, vksh in zip(axs.reshape(-1), v_ref.T, v_ref_shaped.T):
            ax.plot(t, vk)
            ax.plot(t_shaped, vksh, ls='--')
        plt.show()

    print(f"tf_{shaper} = {t_shaped[-1]}")

    if save:
        dq_ref_shaped = np.pad(dq_shaped, ((100,100), (0,0)), mode='constant')
        q_ref_shaped = np.pad(q_shaped[::10,:], ((100,300), (0,0)), mode='edge')
        np.savetxt('js_opt_traj_shaped.csv', dq_ref_shaped,  fmt='%.20f', delimiter=',')
        np.savetxt('js_opt_traj_4rviz.csv', q_ref_shaped,  fmt='%.20f', delimiter=',')

    return t_shaped, dq_shaped, v_ref_shaped


def percentage_of_residual_vibr(wn, zeta, t, A):
    """ Computer percentage of residual vibrations
    """
    wd = wn*np.sqrt(1 - zeta**2)
    C = np.array([Ai*np.exp(zeta*wn*ti)*np.cos(wd*ti) for ti, Ai in zip(t,A)])
    S = np.array([Ai*np.exp(zeta*wn*ti)*np.sin(wd*ti) for ti, Ai in zip(t,A)])
    V = np.sqrt(C.sum()**2 + S.sum()**2)
    return V


if __name__ == "__main__":
    wn = 17.95
    zeta = 0.01
    ts = 0.001

    t_zv, A_zv = ZV_input_shaper(wn, zeta, ts)
    t_umzv, A_umzv = UM_ZV_input_shaper(wn, zeta, ts)
    t_pszv, A_pszv = PS_ZV_input_shaper(wn, zeta, ts)

    V_zv = percentage_of_residual_vibr(wn, zeta, t_zv, A_zv)
    V_umzv = percentage_of_residual_vibr(wn, zeta, t_umzv, A_umzv)
    V_pszv = percentage_of_residual_vibr(wn, zeta, t_pszv, A_pszv)

    print(f"ZV input shaper: V={V_zv:.5f}, T = {t_zv[-1]:.3f}")
    print(f"UM ZV input shaper: V={V_umzv:.5f}, T = {t_umzv[-1]:.3f}")
    print(f"PS ZV input shaper: V={V_pszv:.5f}, T = {t_pszv[-1]:.3f}")