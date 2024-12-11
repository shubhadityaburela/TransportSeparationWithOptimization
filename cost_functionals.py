from minimizer_helper import generate_phi, Q_recons
from preliminaries import *


# ============================================================================ #
#                     FUNCTIONS FOR COST FUNCTIONAL                            #
# ============================================================================ #
def H(Q, alpha, beta, param):
    phiBeta = generate_phi(param, beta)
    return np.linalg.norm(Q - Q_recons(phiBeta, alpha), ord='fro') ** 2


def g(alpha, param):
    return param.alpha_solver_lamda_1 * sum([np.linalg.norm(alpha[param.K_st[nf]:param.K_st[nf + 1], :],
                                                            ord=1) for nf in range(param.nf)])


def f(beta, param):
    return param.beta_solver_lamda_2 * sum([np.linalg.norm(beta[param.degree_st[nf]:param.degree_st[nf + 1]],
                                                           ord=1) for nf in range(param.nf)])


def f_tilde():
    return 0


def J(Q, alpha, beta, param):
    """
    Compute the value of the objective function J
    """
    return H(Q, alpha, beta, param) + g(alpha, param) + f(beta, param) + f_tilde()