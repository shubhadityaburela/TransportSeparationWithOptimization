import numpy as np
import opt_einsum as oe
from preliminaries import *


# ============================================================================ #
#                    Auxiliary Functions and Structures                        #
# ============================================================================ #
def shift(param, beta, nf):
    if param.type_shift[nf] == 1:  # 1 for polynomial
        return numba_polyval(beta, param.t)
    elif param.type_shift[nf] == 2:   # 2 for Sine
        print("Sine shift not implemented yet")
        exit()
    elif param.type_shift[nf] == 3:   # for polynomial + Sine
        print("Polynomial and sine shift mixture not implemented yet")
        exit()

def basis(param, shift, nf):
    if param.type_basis[nf] == 1:  # 1 for Gaussian basis
        X, S = np.meshgrid(param.x, shift)
        return gaussian(X.T, S.T, param.sigma[nf])
    elif param.type_basis[nf] == 2:   # 2 for Cubic basis
        X, S = np.meshgrid(param.x, shift)
        return cubic(X.T, S.T)
    else:
        print("Other basis not implemented yet")
        exit()


def facJac(param, shift, nf, d):
    if param.type_basis[nf] == "gaussian":
        X, S = np.meshgrid(param.x, shift)
        pre = basis(param, shift, nf) * gaussian_exponent(X.T, S.T, param.sigma[nf])
        if param.type_shift[nf] == "polynomial":
            return pre * (param.t ** (param.degree[nf] - d - 1)).reshape(1, -1)
        elif param.type_shift[nf] == "sine":
            print("Sine shift not implemented yet")
            exit()
        elif param.type_shift[nf] == "polynomial+sine":
            print("Polynomial and sine shift mixture not implemented yet")
            exit()
    elif param.type_basis[nf] == "cubic":
        X, S = np.meshgrid(param.x, shift)
        pre = cubic_der(X.T, S.T)
        if param.type_shift[nf] == "polynomial":
            return pre * (param.t ** (param.degree[nf] - d - 1)).reshape(1, -1)
        elif param.type_shift[nf] == "sine":
            print("Sine shift not implemented yet")
            exit()
        elif param.type_shift[nf] == "polynomial+sine":
            print("Polynomial and sine shift mixture not implemented yet")
            exit()
    else:
        print("Other basis not implemented yet")
        exit()


def generate_phi(param, beta):
    phiBeta = np.zeros((param.nf, param.n, param.m))
    for nf in range(param.nf):
        beta_f = beta[param.degree_st[nf]:param.degree_st[nf + 1]]
        shift_f = param.center_of_matrix[nf] + numba_polyval(beta_f, param.t)
        X, S = np.meshgrid(param.x, shift_f)
        phiBeta[nf, ...] = gaussian(X.T, S.T, param.sigma[nf])

    return phiBeta


def generate_phiJac(param, beta):
    phiBeta = np.zeros((param.nf, param.n, param.m))
    phiBetaJac = np.zeros((sum(param.degree), param.n, param.m))
    deg = 0
    for nf in range(param.nf):
        beta_f = beta[param.degree_st[nf]:param.degree_st[nf + 1]]
        shift_f = param.center_of_matrix[nf] + numba_polyval(beta_f, param.t)
        X, S = np.meshgrid(param.x, shift_f)
        phiBeta[nf, ...] = gaussian(X.T, S.T, param.sigma[nf])
        pre = phiBeta[nf, ...] * gaussian_exponent(X.T, S.T, param.sigma[nf])
        for d in range(param.degree[nf]):
            phiBetaJac[deg, ...] = pre * (param.t ** (param.degree[nf] - d - 1)).reshape(1, -1)
            deg = deg + 1

    return phiBeta, phiBetaJac

@njit
def prox_l1(data, reg_param):
    tmp = np.abs(data) - reg_param
    tmp = (tmp + np.abs(tmp)) / 2
    y = np.sign(data) * tmp
    return y


def Q_recons(phiBeta, alpha):
    # return np.einsum('ijk,ik->jk', phiBeta, alpha, optimize="optimal")
    return oe.contract('ijk,ik->jk', phiBeta, alpha)


def reconstruction_error(Q, alpha, beta, param):
    phiBeta = generate_phi(param, beta)
    return np.linalg.norm(Q - Q_recons(phiBeta, alpha)) / np.linalg.norm(Q)
