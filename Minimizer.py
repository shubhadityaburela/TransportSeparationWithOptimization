# -*- coding: utf-8 -*-
"""
Block Coordinate Descent scheme with proximal algorithm
Skeleton version
"""
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================ #
#                    Auxiliary Functions and Structures                        #
# ============================================================================ #

def gaussian_exponent(x, mu, sigma=3.0):
    return (x - mu) / np.power(sigma, 2.0)


def gaussian(x, mu, sigma=3.0):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


def generate_data(param, beta1, beta2):
    x = np.arange(0, param.n)
    t = np.linspace(-10, 10, param.m)
    [X, T] = np.meshgrid(x, t)
    X = X.T
    T = T.T

    q1 = np.zeros_like(X, dtype=np.float64)
    q2 = np.zeros_like(X, dtype=np.float64)
    shift1 = np.polyval(beta1, t)
    shift2 = np.polyval(beta2, t)
    for col in range(param.m):
        for row in range(param.n):
            q1[row, col] = gaussian(row, param.center_of_matrix1 + shift1[col])
            q2[row, col] = gaussian(row, param.center_of_matrix2 + shift2[col])

    Q = np.maximum(q1, q2)

    return Q, q1, q2, x, t


def generate_phi(param, beta1, beta2):
    t = np.linspace(-10, 10, param.m)
    phi1Beta1 = np.zeros((param.n, param.m, param.K_1))
    phi2Beta2 = np.zeros((param.n, param.m, param.K_2))
    shift1 = np.polyval(beta1, t)
    shift2 = np.polyval(beta2, t)
    for col in range(param.m):
        for row in range(param.n):
            phi1Beta1[row, col, 0] = gaussian(row, param.center_of_matrix1 + shift1[col])
            phi2Beta2[row, col, 0] = gaussian(row, param.center_of_matrix2 + shift2[col])

    return phi1Beta1, phi2Beta2


def prox_l1(data, reg_param):
    tmp = abs(data) - reg_param
    tmp = (tmp + abs(tmp)) / 2
    y = np.sign(data) * tmp
    return y


def reconstruction_error(Q, alpha, beta, param):
    beta1 = beta[:param.degree1]
    beta2 = beta[param.degree1:]
    phi1Beta1, phi2Beta2 = generate_phi(param, beta1, beta2)

    return np.linalg.norm(Q - Q_recons(phi1Beta1, phi2Beta2, alpha, param)) / np.linalg.norm(Q)


# ============================================================================ #
#                     FUNCTIONS FOR COST FUNCTIONAL                            #
# ============================================================================ #
def Q_recons(phi1Beta1, phi2Beta2, alpha, param):
    Q = np.einsum('ijk,kj->ij', phi1Beta1, alpha[:param.K_1]) + np.einsum('ijk,kj->ij', phi2Beta2, alpha[param.K_1:],
                                                                          optimize="optimal")
    return Q


def H(Q, alpha, beta, param):
    beta1 = beta[:param.degree1]
    beta2 = beta[param.degree1:]
    phi1Beta1, phi2Beta2 = generate_phi(param, beta1, beta2)

    return np.linalg.norm(Q - Q_recons(phi1Beta1, phi2Beta2, alpha, param), ord='fro') ** 2


def g(alpha, param):
    return param.alpha_solver_lamda_1 * (np.linalg.norm(alpha[:param.K_1, :], ord=1)
                                         + np.linalg.norm(alpha[param.K_1:, :], ord=1))


def f(beta, param):
    return param.beta_solver_lamda_2 * (np.linalg.norm(beta[:param.degree1], ord=1)
                                        + np.linalg.norm(beta[param.degree1:], ord=1))


def f_tilde():
    return 0


def J(Q, alpha, beta, param):
    """
    Compute the value of the objective function J
    """
    return H(Q, alpha, beta, param) + g(alpha, param) + f(beta, param) + f_tilde()


# ============================================================================ #
#                     PROXIMAL STEP FOR ALPHA UPDATE                           #
# ============================================================================ #
def gradient_H_alpha(Q, alpha, beta, param):
    """
    Compute the gradient of the least-squares term H in J w.r.t alpha.
    """
    # phi => [R^{n X m X K_1}, R^{n X m X K_2}]  # We have 2 frames (nf) and phi is a list of 2 3-tensors
    beta1 = beta[:param.degree1]
    beta2 = beta[param.degree1:]
    phi1Beta1, phi2Beta2 = generate_phi(param, beta1, beta2)

    R = Q - Q_recons(phi1Beta1, phi2Beta2, alpha, param)
    dH_dAlpha_1 = - 2 * np.einsum('ij,ijk->kj', R, phi1Beta1, optimize="optimal")
    dH_dAlpha_2 = - 2 * np.einsum('ij,ijk->kj', R, phi2Beta2, optimize="optimal")

    return np.concatenate((dH_dAlpha_1, dH_dAlpha_2), axis=0)


def argmin_H_alpha(Q, alpha, beta, param):
    """
    Minimize J w.r.t alpha using a single step forward-backward
    """
    ct = param.alpha_solver_lamda_1 / param.alpha_solver_ck
    return prox_l1(alpha - (1 / param.alpha_solver_ck) * gradient_H_alpha(Q, alpha, beta, param), ct)


# ============================================================================ #
#                 PROXIMAL PRIMAL-DUAL STEP FOR BETA UPDATE                    #
# ============================================================================ #
def generate_phiJacLinear(param, beta1, beta2):
    t = np.linspace(-10, 10, param.m)
    phi1Beta1 = np.zeros((param.n, param.m, param.K_1))
    phi2Beta2 = np.zeros((param.n, param.m, param.K_2))
    phi1Beta1Jac = np.zeros((param.n, param.m, param.K_1, param.degree1))
    phi2Beta2Jac = np.zeros((param.n, param.m, param.K_2, param.degree2))
    shift1 = np.polyval(beta1, t)
    shift2 = np.polyval(beta2, t)
    for col in range(param.m):
        for row in range(param.n):
            phi1Beta1[row, col, 0] = gaussian(row, param.center_of_matrix1 + shift1[col])
            phi2Beta2[row, col, 0] = gaussian(row, param.center_of_matrix2 + shift2[col])

            exponent1 = gaussian_exponent(row, param.center_of_matrix1 + shift1[col])
            phi1Beta1Jac[row, col, 0, 0] = phi1Beta1[row, col, 0] * exponent1 * t[col]
            phi1Beta1Jac[row, col, 0, 1] = phi1Beta1[row, col, 0] * exponent1

            exponent2 = gaussian_exponent(row, param.center_of_matrix2 + shift2[col])
            phi2Beta2Jac[row, col, 0, 0] = phi2Beta2[row, col, 0] * exponent2 * t[col]
            phi2Beta2Jac[row, col, 0, 1] = phi2Beta2[row, col, 0] * exponent2


    return phi1Beta1, phi2Beta2, phi1Beta1Jac, phi2Beta2Jac



def generate_phiJacQuadPolynomial(param, beta1, beta2):
    t = np.linspace(-10, 10, param.m)
    phi1Beta1 = np.zeros((param.n, param.m, param.K_1))
    phi2Beta2 = np.zeros((param.n, param.m, param.K_2))
    phi1Beta1Jac = np.zeros((param.n, param.m, param.K_1, param.degree1))
    phi2Beta2Jac = np.zeros((param.n, param.m, param.K_2, param.degree2))
    shift1 = np.polyval(beta1, t)
    shift2 = np.polyval(beta2, t)
    for col in range(param.m):
        for row in range(param.n):
            phi1Beta1[row, col, 0] = gaussian(row, param.center_of_matrix1 + shift1[col])
            phi2Beta2[row, col, 0] = gaussian(row, param.center_of_matrix2 + shift2[col])

            exponent1 = gaussian_exponent(row, param.center_of_matrix1 + shift1[col])
            phi1Beta1Jac[row, col, 0, 0] = phi1Beta1[row, col, 0] * exponent1 * t[col]
            phi1Beta1Jac[row, col, 0, 1] = phi1Beta1[row, col, 0] * exponent1

            exponent2 = gaussian_exponent(row, param.center_of_matrix2 + shift2[col])
            phi2Beta2Jac[row, col, 0, 0] = phi2Beta2[row, col, 0] * exponent2 * (t[col]) ** 2
            phi2Beta2Jac[row, col, 0, 1] = phi2Beta2[row, col, 0] * exponent2 * t[col]
            phi2Beta2Jac[row, col, 0, 2] = phi2Beta2[row, col, 0] * exponent2


    return phi1Beta1, phi2Beta2, phi1Beta1Jac, phi2Beta2Jac


def gradient_H_beta(Q, alpha, beta, param):
    """
    Compute the gradient of the least-squares term H in J w.r.t beta
    """
    beta1 = beta[:param.degree1]
    beta2 = beta[param.degree1:]
    phi1Beta1, phi2Beta2, Phi1Beta1Jac, Phi2Beta2Jac = generate_phiJacLinear(param, beta1, beta2)

    R = Q - Q_recons(phi1Beta1, phi2Beta2, alpha, param)

    dH_dBeta_1 = -2 * np.einsum('ij,ijkl,kj->l', R, Phi1Beta1Jac, alpha[:param.K_1], optimize="optimal")
    dH_dBeta_2 = -2 * np.einsum('ij,ijkl,kj->l', R, Phi2Beta2Jac, alpha[param.K_1:], optimize="optimal")

    return np.concatenate((dH_dBeta_1, dH_dBeta_2), axis=0)


def argmin_H_beta(Q, alpha, beta, param):
    """
    Minimize J w.r.t beta
    """
    # Initialization
    u = np.zeros_like(beta)  # Dual variable (Same shape as the beta variable)
    crit = np.Inf  # Initial value of the objective function
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
        # if n > param.beta_solver_maxit // 10:
        #     eta = eta / 2
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
    # beta => R^{degree1 + degree2}
    # alpha => R^{(K_1 + K_2) X m}
    beta = np.hstack((param.beta1_init, param.beta2_init))
    alpha = np.zeros((param.K_1 + param.K_2, param.m))

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

