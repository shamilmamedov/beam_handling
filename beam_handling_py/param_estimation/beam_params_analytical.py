#!/usr/bin/env python3

import casadi as cs
from scipy import integrate


""" Compute natural frequency based on lumping of the flexible link into
a mass and a spring. The consitency between the lumped and original model
is based on constrained assumed mode method
"""

def area_moment_inertia(h, b, d):
    """ Area moment of inertia for a rectangular
    see https://youtu.be/K2fKTKdaVGw for details
    """
    A = h*b
    Ic = b*h**3/12
    return Ic + A*d**2

if __name__ == "__main__":
    L = 52e-2 # 52e-2
    w = 8e-2
    h = 1e-3
    E = 1.9e11 # 190 to 205 Gpa
    rho = 7.87e3 # 7.87 to 8.07 kg/m3

    Volume = h*L*w
    Area = h*w
    m = Volume*rho
    mu = Area*rho

    I = area_moment_inertia(h, w, 0)
    print(f"Area moment of inertia = {I:e}")
    print(f"Fluxural rigidity = {E*I:.5f}")
    print(f"Mass = {m:.3f}")

    # Create functions for computing assumed mode shapes
    # A function to find betas
    beta = cs.MX.sym("beta", 1)
    beta_expr = 1 + cs.cosh(beta)*cs.cos(beta)
    beta_fcn = cs.Function("beta_fcn", [beta], [beta_expr])

    # Plot beta function for providing a good inital guess to rootfinder
    G = cs.rootfinder('G', 'newton', beta_fcn)
    beta1_0 = cs.pi/2
    beta2_0 = 2*cs.pi
    beta1 = float(G(beta1_0))
    beta2 = float(G(beta2_0))
    print(f"\nbeta1 = {beta1:.4f}, beta2 = {beta2:.4f}")
    gamma1 = (cs.cosh(beta1) + cs.cos(beta1))/(cs.sinh(beta1) + cs.sin(beta1))
    gamma2 = (cs.cosh(beta2) + cs.cos(beta2))/(cs.sinh(beta2) + cs.sin(beta2))
    print(f"gamma1 = {gamma1:.4f}, gamma2 = {gamma2:.4f}")
    A = cs.sqrt(L)

    # Mode shape functions
    x = cs.MX.sym("x", 1)
    F1_expr = 1/A*(cs.cosh(beta1/L*x) - cs.cos(beta1/L*x) - gamma1*(cs.sinh(beta1/L*x) - cs.sin(beta1/L*x)))
    F1_fcn = cs.Function("F1_fcn", [x], [F1_expr])

    # Find lumped mass
    fcn = lambda x: float(F1_fcn(x))**2
    Me_lin = mu*integrate.quad(fcn, 0, L)[0]/float(F1_fcn(L))**2
    Me_rot = mu*integrate.quad(fcn, 0, L)[0]/L**2
    print(f"\nLumped linear mass = {Me_lin:.5f}")
    print(f"Lumped rotational mass = {Me_rot:.5f}")

    # Find lumped stiffness
    F1_hess_expr = cs.hessian(F1_expr, x)[0]
    F1_hess_fcn = cs.Function("F1_hess_fcn", [x], [F1_hess_expr])

    fcn = lambda x: float(F1_hess_fcn(x))**2
    Ke_lin = E*I*integrate.quad(fcn, 0, L)[0]/float(F1_fcn(L))**2
    Ke_rot = E*I*integrate.quad(fcn, 0, L)[0]
    print(f"\nLumped linear stiffness = {Ke_lin:.5f}")
    print(f"Lumped rotational stiffness = {Ke_rot:.5f}")
    print(f"\nNatural frequency = {cs.sqrt(Ke_lin/Me_lin):.4f}")
    print(f"Natural frequency upward = {cs.sqrt(Ke_lin/Me_lin - 9.81/L):.4f}")
    print(f"Natural frequency downward = {cs.sqrt(Ke_lin/Me_lin + 9.81/L):.4f}")
