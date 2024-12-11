# -*- coding: utf-8 -*-
"""
Block Coordinate Descent scheme with proximal algorithm
Skeleton version
"""
import numpy as np
import opt_einsum as oe

from cost_functionals import J
from minimizer_helper import generate_phi, Q_recons, prox_l1, generate_phiJac, reconstruction_error


# ============================================================================ #
#                     PROXIMAL STEP FOR ALPHA UPDATE                           #
# ============================================================================ #
def gradient_H_alpha(Q, alpha, beta, param):
    """
    Compute the gradient of the least-squares term H in J w.r.t alpha.
    """
    # phi => R^{nf X n X m}
    # We have nf frames and phi 3-tensor with nf being the first dimension
    phiBeta = generate_phi(param, beta)

    R = Q - Q_recons(phiBeta, alpha)
    # dH_dAlpha = - 2 * np.einsum('jk,ijk->ik', R, phiBeta, optimize="optimal")
    dH_dAlpha = - 2 * oe.contract('jk,ijk->ik', R, phiBeta)

    return dH_dAlpha


def argmin_H_alpha(Q, alpha, beta, param):
    """
    Minimize J w.r.t alpha using a single step forward-backward
    """
    ct = param.alpha_solver_lamda_1 / param.alpha_solver_ck
    return prox_l1(alpha - (1 / param.alpha_solver_ck) * gradient_H_alpha(Q, alpha, beta, param), ct)


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

    # dH_dBeta= -2 * np.einsum('jk,ijk,lk->i', R, phiBetaJac, alpha, optimize="optimal")
    dH_dBeta = -2 * oe.contract('jk,ijk,lk->i', R, phiBetaJac, alpha)
    return dH_dBeta


def argmin_H_beta(Q, alpha, beta, param):
    """
    Minimize J w.r.t beta
    """
    # Initialization
    u = np.zeros_like(beta)  # Dual variable (Same shape as the beta variable)
    crit = np.inf  # Initial value of the objective function
    eta = param.beta_solver_rho_n

    # Main loop
    print("------------------------------")
    for n in range(param.beta_solver_maxit):
        print(f"Beta iteration counter : {n}")
        # 1) Primal update
        beta_half = beta - param.beta_solver_tau * gradient_H_beta(Q, alpha, beta, param) - param.beta_solver_tau * u
        # beta_half[beta_half < 0] = 0  # Proximal operator is equal to the projection for indicator function

        # 2) Dual update
        uTemp = u + (2 * beta_half - beta)
        u_half = uTemp - prox_l1(uTemp, param.beta_solver_lamda_2 / param.beta_solver_sigma)

        # 3) Inertial update
        beta = beta + eta * (beta_half - beta)
        u = u + eta * (u_half - u)

        # 4) Check stopping criterion (convergence in terms of objective function)
        crit_old = crit
        crit = J(Q, alpha, beta, param)
        if np.abs(crit_old - crit) < param.beta_solver_gtol * crit:
            print(f"Beta loop converged after {n} iterations")
            break

    return beta


# ============================================================================ #
#     ALTERNATING MINIMIZATION WITH ALPHA AND BETA IN BLOCK CD TYPE (PALM)     #
# ============================================================================ #
def argmin_H(Q, param):
    """
    Minimize J w.r.t alpha and beta using block coordinate descent scheme (refer PALM algorithm)
    """

    # Save the cost functional and relative reconstruction error
    J_list = []
    recerr_list = []

    obj_val = np.inf

    # Initialize alpha, beta
    # beta => R^{degree[1] + degree[2] + .....}
    # alpha => R^{(K[1] + K[2] + ......) X m}
    beta = np.hstack(param.beta_init)
    alpha = np.zeros((sum(param.K), param.m))

    for it in range(param.maxit):

        print("\n**************************************************************")
        print(f"Outer iteration counter : {it}")

        # Optimize over alpha
        alpha = argmin_H_alpha(Q, alpha, beta, param)

        # Optimize over beta
        beta = argmin_H_beta(Q, alpha, beta, param)

        old_obj = obj_val
        obj_val = J(Q, alpha, beta, param)
        rec_err = reconstruction_error(Q, alpha, beta, param)

        J_list.append(obj_val)
        recerr_list.append(rec_err)
        print(beta)
        print(f"J : {obj_val}, RecErr : {rec_err}")

        if abs(obj_val - old_obj) < param.gtol * old_obj:
            print(f"Outer loop converged after {it} iterations")
            break

    return alpha, beta, J_list, recerr_list
