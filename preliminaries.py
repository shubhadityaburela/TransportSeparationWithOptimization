import numpy as np


# ============================================================================ #
#                       Type of basis functions and shifts                     #
# ============================================================================ #

def gaussian_exponent(x, mu, sigma=3.0):
    return (x - mu) / np.power(sigma, 2.0)


def gaussian(x, mu, sigma=3.0):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


def polynomial(c, t):
    return np.polyval(c, t)


def cubic(x, mu):
    return (x - mu) ** 3


def cubic_der(x, mu):
    return 3 * ((x - mu) ** 2)

