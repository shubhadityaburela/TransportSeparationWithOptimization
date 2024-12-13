import numpy as np
from numba import njit, prange
import torch


# ============================================================================ #
#               Type of basis functions and shifts (TORCH)                     #
# ============================================================================ #

@torch.jit.script
def torch_gaussian_exponent(x, mu, sigma):
    return (x - mu) / torch.pow(sigma, 2.0)


@torch.jit.script
def torch_gaussian(x, mu, sigma):
    return torch.exp(-torch.pow(x - mu, 2.0) / (2 * torch.pow(sigma, 2.0)))


@torch.jit.script
def torch_polyval(coeffs, x_array):
    results = torch.zeros_like(x_array)
    for i in range(x_array.shape[0]):
        x = x_array[i]
        result = 0.0
        for coeff in coeffs:
            result = result * x + coeff
        results[i] = result
    return results





# ============================================================================ #
#               Type of basis functions and shifts (NUMBA)                     #
# ============================================================================ #

@njit
def gaussian_exponent(x, mu, sigma=3.0):
    return (x - mu) / np.power(sigma, 2.0)

@njit
def gaussian(x, mu, sigma=3.0):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))

@njit
def polynomial(c, t):
    return numba_polyval(c, t)

@njit
def cubic(x, mu):
    return (x - mu) ** 3

@njit
def cubic_der(x, mu):
    return 3 * ((x - mu) ** 2)


@njit(parallel=True)
def numba_polyval(coeffs, x_array):
    """
    Evaluate a polynomial at multiple points using Horner's method.

    Parameters:
        coeffs (array-like): Coefficients of the polynomial (highest degree first).
        x_array (array-like): Array of points at which to evaluate the polynomial.

    Returns:
        ndarray: Array of results for each point in x_array.
    """
    results = np.zeros_like(x_array)
    for i in prange(x_array.shape[0]):
        x = x_array[i]
        result = 0.0
        for coeff in coeffs:
            result = result * x + coeff
        results[i] = result
    return results


