from preliminaries import *


# ============================================================================ #
#                          Data generation functions                           #
# ============================================================================ #
def generate_data(param, beta):
    param.x = np.arange(0, param.n)
    param.t = np.linspace(param.t_start, param.t_end, param.m)

    q1 = np.zeros((param.n, param.m))
    q2 = np.zeros((param.n, param.m))
    shift1 = np.polyval(beta[0], param.t)
    shift2 = np.polyval(beta[1], param.t)
    for col in range(param.m):
        q1[:, col] = gaussian(param.x, param.center_of_matrix[0] + shift1[col], param.sigma[0][0])
        q2[:, col] = gaussian(param.x, param.center_of_matrix[1] + shift2[col], param.sigma[1][0])

    Q = np.maximum(q1, q2)

    return Q, q1, q2


def generate_data_faded(param, beta):
    param.x = np.arange(0, param.n)
    param.t = np.linspace(param.t_start, param.t_end, param.m)

    # Construct the fading of the waves
    damp = np.arange(len(param.t))
    damp1 = damp[::-1] / np.max(damp) + 0.3
    damp2 = (damp / (np.max(damp) / 2)) + 2.5

    q1 = np.zeros((param.n, param.m))
    q2 = np.zeros((param.n, param.m))
    shift1 = np.polyval(beta[0], param.t)
    shift2 = np.polyval(beta[1], param.t)
    for col in range(param.m):
        sigma_t = 1.5
        q1[:, col] = gaussian(param.x, param.center_of_matrix[0] + shift1[col], sigma_t * damp1[col])
        q2[:, col] = gaussian(param.x, param.center_of_matrix[1] + shift2[col], sigma_t * damp2[col])

    Q = np.maximum(q1, q2)

    return Q, q1, q2


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

