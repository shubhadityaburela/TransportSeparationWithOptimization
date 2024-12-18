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


def generate_phi(params, beta):
    X, MU = torch.meshgrid(params.x, params.center_of_matrix + torch_polyval(beta, params.t))
    return torch_gaussian(X, MU, params.sigma)


def generate_phiJac(params, beta):
    phiBetaJac = torch.zeros((params.degree, params.n, params.m), dtype=torch.float32, device=params.device)
    X, MU = torch.meshgrid(params.x, params.center_of_matrix + torch_polyval(beta, params.t))
    phiBeta = torch_gaussian(X, MU, params.sigma)
    pre = phiBeta * torch_gaussian_exponent(X, MU, params.sigma)
    deg = 0
    for d in range(params.degree):
        phiBetaJac[deg, ...] = pre * (params.t ** (params.degree - d - 1)).reshape(1, -1)
        deg = deg + 1

    return phiBeta, phiBetaJac


@torch.jit.script
def prox_l1(data, reg_param):
    tmp = torch.abs(data) - reg_param
    tmp = (tmp + torch.abs(tmp)) / 2
    y = torch.sign(data) * tmp
    return y


@torch.jit.script
def Q_recons(phiBeta, alpha):
    return torch.einsum('ij,j->ij', phiBeta, alpha)


def reconstruction_error(Q, phiBeta, alpha):
    return torch.linalg.norm(Q - Q_recons(phiBeta, alpha)) / torch.linalg.norm(Q)
