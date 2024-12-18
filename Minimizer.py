# -*- coding: utf-8 -*-
"""
Block Coordinate Descent scheme with proximal algorithm
Skeleton version
"""
import torch

from cost_functionals import J
from minimizer_helper import Q_recons, prox_l1


# ============================================================================ #
#                     PROXIMAL STEP FOR ALPHA UPDATE                           #
# ============================================================================ #
def gradient_H_alpha(Q, phiBeta, alpha):
    """
    Compute the gradient of the least-squares term H in J w.r.t alpha.
    """
    R = Q - Q_recons(phiBeta, alpha)
    dH_dAlpha = - 2 * torch.einsum('ij,ij->j', R, phiBeta)

    return dH_dAlpha


def argmin_H_alpha(Q, phiBeta, alpha, params):
    """
    Minimize J w.r.t alpha using a single step forward-backward
    """
    ct = params.alpha_solver_lamda_1 / params.alpha_solver_ck
    return prox_l1(alpha - (1 / params.alpha_solver_ck) * gradient_H_alpha(Q, phiBeta, alpha), ct)


# ============================================================================ #
#                 PROXIMAL PRIMAL-DUAL STEP FOR BETA UPDATE                    #
# ============================================================================ #
def gradient_H_beta(Q, phiBeta, phiBetaJac, alpha):
    """
    Compute the gradient of the least-squares term H in J w.r.t beta
    """
    # phi => R^{nf X n X m}
    # We have nf frames and phi 3-tensor with nf being the first dimension

    # phiBetaJac => [R^{sum(degree[1st frame] + degree[2nd frame] + ...) X n X m}]
    R = Q - Q_recons(phiBeta, alpha)

    dH_dBeta = -2 * torch.einsum('ij,ijk,j->k', R, phiBetaJac, alpha)
    return dH_dBeta


def argmin_H_beta(Q, phiBeta, phiBetaJac, alpha, beta, params):
    """
    Minimize J w.r.t beta
    """
    # Initialization
    u = torch.zeros_like(beta)  # Dual variable (Same shape as the beta variable)
    crit = torch.inf  # Initial value of the objective function
    eta = params.beta_solver_rho_n

    # Main loop
    for n in range(params.beta_solver_maxit):
        # 1) Primal update
        beta_half = beta - params.beta_solver_tau * gradient_H_beta(Q, phiBeta, phiBetaJac, alpha) - params.beta_solver_tau * u

        # 2) Dual update
        uTemp = u + (2 * beta_half - beta)
        u_half = uTemp - prox_l1(uTemp, params.beta_solver_lamda_2 / params.beta_solver_sigma)

        # 3) Inertial update
        beta = beta + eta * (beta_half - beta)
        u = u + eta * (u_half - u)

        # 4) Check stopping criterion (convergence in terms of objective function)
        crit_old = crit
        crit = J(Q, phiBeta, alpha, beta, params)
        if torch.abs(crit_old - crit) < params.beta_solver_gtol * crit:
            print(f"Tensor optimization for beta.... Beta loop converged after {n} iterations")
            break

    return beta


# ============================================================================ #
#     ALTERNATING MINIMIZATION WITH ALPHA AND BETA IN BLOCK CD TYPE (PALM)     #
# ============================================================================ #
def argmin_H(Q, phiBeta, phiBetaJac, alpha, beta, params):
    """
    Minimize J w.r.t alpha and beta using block coordinate descent scheme (refer PALM algorithm)
    """
    print("------------------------------")

    # Optimize over alpha
    alpha = argmin_H_alpha(Q, phiBeta, alpha, params)

    # Optimize over beta
    beta = argmin_H_beta(Q, phiBeta, phiBetaJac, alpha, beta, params)
    print("------------------------------")

    return alpha, beta
