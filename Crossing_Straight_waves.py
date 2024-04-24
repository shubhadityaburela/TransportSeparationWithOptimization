import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from math import sqrt
from scipy.sparse import diags
from scipy.linalg import cholesky
from dataclasses import dataclass
from matplotlib.ticker import MaxNLocator
import os

import matplotlib
matplotlib.use('TkAgg')


from Minimizer import generate_data, argmin_H, generate_phi


@dataclass
class Parameters:
    nf: int = 2
    n: int = 150
    m: int = 75
    sigma: float = 1.0
    K_1: int = 1
    K_2: int = 1
    degree1: int = 2
    degree2: int = 2

    beta1_init = [1, -1]
    beta2_init = [-1, 1]
    beta1 = [4, -1]
    beta2 = [-3, -3]
    center_of_matrix1: float = 75
    center_of_matrix2: float = 75

    alpha_solver_ck: float = 1000
    alpha_solver_lamda_1: float = 0.05

    beta_solver_tau: float = 0.001
    beta_solver_sigma: float = 0.99 / beta_solver_tau
    beta_solver_lamda_2: float = 0.05
    beta_solver_rho_n: float = 1
    beta_solver_gtol: float = 1e-3
    beta_solver_maxit: int = 50

    gtol: float = 1e-5
    maxit: int = 2000


if __name__ == '__main__':

    # Instantiate the constants for the optimization
    param = Parameters()

    # Generate the data
    Q, _, _, x, t = generate_data(param, param.beta1, param.beta2)

    # Optimize over alpha and beta
    alpha, beta, J, R = argmin_H(Q, param)

    # Reconstruct the individual frames after separation and convergence
    beta1 = beta[:param.degree1]
    beta2 = beta[param.degree1:]
    phi1Beta1, phi2Beta2 = generate_phi(param, beta1, beta2)
    Q1 = np.einsum('ijk,kj->ij', phi1Beta1, alpha[:param.K_1], optimize="optimal")
    Q2 = np.einsum('ijk,kj->ij', phi2Beta2, alpha[param.K_1:], optimize="optimal")

    # Plots the results
    impath = "plots/Straight/"  # For plots
    os.makedirs(impath, exist_ok=True)

    # Plot the separated frames
    fig, axs = plt.subplots(1, 4, figsize=(16, 6))
    vmin = np.min(Q)
    vmax = np.max(Q)
    # Original
    axs[0].imshow(Q, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
    axs[0].set_title('Original')
    axs[0].set_xlabel('$t_j$')
    axs[0].set_ylabel('$x_i$')
    # Frame 1
    axs[1].imshow(Q1, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
    axs[1].set_title('Frame 1')
    axs[1].set_xlabel('$t_j$')
    axs[1].set_ylabel('$x_i$')
    # Frame 2
    axs[2].imshow(Q2, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
    axs[2].set_title('Frame 2')
    axs[2].set_xlabel('$t_j$')
    axs[2].set_ylabel('$x_i$')
    # Reconstructed
    axs[3].imshow(Q1 + Q2, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
    axs[3].set_title('Reconstructed')
    axs[3].set_xlabel('$t_j$')
    axs[3].set_ylabel('$x_i$')
    fig.savefig(impath + "Q", dpi=300, transparent=True)

    # Plot the cost functional
    fig1 = plt.figure(figsize=(12, 12))
    ax1 = fig1.add_subplot(111)
    ax1.semilogy(np.arange(len(J)), J, color="C0", label=r"$J(\alpha, \beta)$")
    ax1.set_xlabel(r"$n_{\mathrm{iter}}$")
    ax1.set_ylabel(r"$J$")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.legend()
    fig1.savefig(impath + "J", dpi=300, transparent=True)