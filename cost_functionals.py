from minimizer_helper import generate_phi, Q_recons
from preliminaries import *


# ============================================================================ #
#                     FUNCTIONS FOR COST FUNCTIONAL                            #
# ============================================================================ #
def H(Q, phiBeta, alpha, beta, params):
    return torch.linalg.norm(Q - Q_recons(phiBeta, alpha), ord='fro') ** 2


def g(alpha, params):
    return params.alpha_solver_lamda_1 * torch.linalg.norm(alpha, ord=1)


def f(beta, params):
    return params.beta_solver_lamda_2 * torch.linalg.norm(beta, ord=1)


def f_tilde():
    return 0


def J(Q, phiBeta, alpha, beta, param):
    """
    Compute the value of the objective function J
    """
    return H(Q, phiBeta, alpha, beta, param) + g(alpha, param) + f(beta, param) + f_tilde()