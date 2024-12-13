# -*- coding: utf-8 -*-
"""
Block Coordinate Descent scheme with proximal algorithm
Skeleton version
"""
import numpy as np
import opt_einsum as oe
import torch

from cost_functionals import J
from minimizer_helper import generate_phi, Q_recons, prox_l1, generate_phiJac, reconstruction_error


# ============================================================================ #
#                     PROXIMAL STEP FOR ALPHA UPDATE                           #
# ============================================================================ #
def gradient_H_alpha(Q, alpha, beta, params):
    """
    Compute the gradient of the least-squares term H in J w.r.t alpha.
    """
    # phi => R^{nf X n X m}
    # We have nf frames and phi 3-tensor with nf being the first dimension
    phiBeta = generate_phi(params, beta)

    R = Q - Q_recons(phiBeta, alpha)
    dH_dAlpha = - 2 * torch.einsum('ij,ij->j', R, phiBeta)

    return dH_dAlpha


def argmin_H_alpha(Q, alpha, beta, params):
    """
    Minimize J w.r.t alpha using a single step forward-backward
    """
    ct = params.alpha_solver_lamda_1 / params.alpha_solver_ck
    return prox_l1(alpha - (1 / params.alpha_solver_ck) * gradient_H_alpha(Q, alpha, beta, params), ct)


# ============================================================================ #
#                 PROXIMAL PRIMAL-DUAL STEP FOR BETA UPDATE                    #
# ============================================================================ #
def gradient_H_beta(Q, alpha, beta, param):
    """
    Compute the gradient of the least-squares term H in J w.r.t beta
    """
    # phi => R^{nf X n X m}
    # We have nf frames and phi 3-tensor with nf being the first dimension

    # phiJac => [R^{sum(degree[1st frame] + degree[2nd frame] + ...) X n X m}]
    phiBeta, phiBetaJac = generate_phiJac(param, beta)
    R = Q - Q_recons(phiBeta, alpha)

    dH_dBeta = -2 * torch.einsum('jk,ijk,k->i', R, phiBetaJac, alpha)
    return dH_dBeta


def argmin_H_beta(Q, alpha, beta, params):
    """
    Minimize J w.r.t beta
    """
    # Initialization
    u = torch.zeros_like(beta)  # Dual variable (Same shape as the beta variable)
    crit = torch.inf  # Initial value of the objective function
    eta = params.beta_solver_rho_n

    # Main loop
    print("------------------------------")
    for n in range(params.beta_solver_maxit):
        print(f"Beta iteration counter : {n}")
        # 1) Primal update
        beta_half = beta - params.beta_solver_tau * gradient_H_beta(Q, alpha, beta, params) - params.beta_solver_tau * u

        # 2) Dual update
        uTemp = u + (2 * beta_half - beta)
        u_half = uTemp - prox_l1(uTemp, params.beta_solver_lamda_2 / params.beta_solver_sigma)

        # 3) Inertial update
        beta = beta + eta * (beta_half - beta)
        u = u + eta * (u_half - u)

        # 4) Check stopping criterion (convergence in terms of objective function)
        crit_old = crit
        crit = J(Q, alpha, beta, params)
        if torch.abs(crit_old - crit) < params.beta_solver_gtol * crit:
            print(f"Beta loop converged after {n} iterations")
            break

    return beta


# ============================================================================ #
#     ALTERNATING MINIMIZATION WITH ALPHA AND BETA IN BLOCK CD TYPE (PALM)     #
# ============================================================================ #
def argmin_H(Q, params):
    """
    Minimize J w.r.t alpha and beta using block coordinate descent scheme (refer PALM algorithm)
    """

    # Save the cost functional and relative reconstruction error
    J_list = []
    recerr_list = []

    obj_val = torch.inf

    # Initialize alpha, beta
    # beta => R^{degree[1] + degree[2] + .....}
    # alpha => R^{(K[1] + K[2] + ......) X m}
    beta = params.beta_init.clone()
    beta = beta.to(device=params.device)
    alpha = torch.zeros(params.m, dtype=torch.float32, device=params.device)

    for it in range(params.maxit):

        print("\n**************************************************************")
        print(f"Outer iteration counter : {it}")

        # Optimize over alpha
        alpha = argmin_H_alpha(Q, alpha, beta, params)

        # Optimize over beta
        beta = argmin_H_beta(Q, alpha, beta, params)

        old_obj = obj_val
        obj_val = J(Q, alpha, beta, params)
        rec_err = reconstruction_error(Q, alpha, beta, params)

        J_list.append(obj_val)
        recerr_list.append(rec_err)
        print(beta)
        print(f"J : {obj_val}, RecErr : {rec_err}")

        if abs(obj_val - old_obj) < params.gtol * old_obj:
            print(f"Outer loop converged after {it} iterations")
            break

    return alpha, beta, J_list, recerr_list
