from preliminaries import *
from numba import njit, prange
import torch


# ============================================================================ #
#                          Data generation functions (torch)                   #
# ============================================================================ #
def generate_data_singleframe_torch(params):
    shift1 = torch_polyval(params.beta, params.t)
    X, MU = torch.meshgrid(params.x, params.center_of_matrix + shift1)
    q1 = torch_gaussian(X, MU, params.sigma)
    return q1


# ============================================================================ #
#                          Data generation functions (NUMBA)                   #
# ============================================================================ #
@njit(parallel=True)
def generate_data(param, beta):
    q1 = np.zeros((param.n, param.m), dtype=np.float64)
    q2 = np.zeros_like(q1)
    shift1 = numba_polyval(beta[0], param.t)
    shift2 = numba_polyval(beta[1], param.t)
    for col in prange(param.m):
        q1[:, col] = gaussian(param.x, param.center_of_matrix[0] + shift1[col], param.sigma[0])
        q2[:, col] = gaussian(param.x, param.center_of_matrix[1] + shift2[col], param.sigma[1])

    Q = np.maximum(q1, q2)

    return Q, q1, q2


@njit(parallel=True)
def generate_data_singleframe(param, beta):
    q1 = np.zeros((param.n, param.m), dtype=np.float64)
    shift1 = numba_polyval(beta, param.t)
    for col in prange(param.m):
        q1[:, col] = gaussian(param.x, param.center_of_matrix[0] + shift1[col], param.sigma[0])

    return q1


@njit(parallel=True)
def generate_data_faded(param, beta):
    # Construct the fading of the waves
    damp = np.arange(len(param.t), dtype=np.float64)
    damp1 = damp[::-1] / np.max(damp) + 0.3
    damp2 = (damp / (np.max(damp) / 10)) + 2.5

    q1 = np.zeros((param.n, param.m), dtype=np.float64)
    q2 = np.zeros_like(q1)
    shift1 = numba_polyval(beta[0], param.t)
    shift2 = numba_polyval(beta[1], param.t)
    for col in prange(param.m):
        sigma_t = 1.5
        q1[:, col] = gaussian(param.x, param.center_of_matrix[0] + shift1[col], sigma_t * damp1[col])
        q2[:, col] = gaussian(param.x, param.center_of_matrix[1] + shift2[col], sigma_t * damp2[col])

    Q = np.maximum(q1, q2)

    return Q, q1, q2


@njit(parallel=True)
def generate_data_faded_singleframe(param, beta):
    # Construct the fading of the waves
    damp = np.arange(len(param.t), dtype=np.float64)
    damp1 = (damp / (np.max(damp) / 10)) + 2.5

    q1 = np.zeros((param.n, param.m), dtype=np.float64)
    shift1 = numba_polyval(beta, param.t)
    for col in prange(param.m):
        sigma_t = 1.5
        q1[:, col] = gaussian(param.x, param.center_of_matrix[0] + shift1[col], sigma_t * damp1[col])

    return q1



def generate_data_sine(param, beta1, beta2):
    param.x = np.arange(0, param.n)
    param.t = np.linspace(param.t_start, param.t_end, param.m)

    shift1 = beta1[0] * np.sin(beta1[1] * np.pi * param.t)
    shift2 = np.polyval(beta2, param.t)

    q1 = np.zeros((param.n, param.m))
    q2 = np.zeros((param.n, param.m))

    for col in range(param.m):
        q1[:, col] = gaussian(param.x, param.center_of_matrix1 + shift1[col])
        q2[:, col] = gaussian(param.x, param.center_of_matrix2 + shift2[col])

    Q = np.maximum(q1, q2)

    return Q, q1, q2

