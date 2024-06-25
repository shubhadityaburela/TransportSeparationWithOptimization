import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from math import sqrt
from scipy.sparse import diags
from scipy.linalg import cholesky
from dataclasses import dataclass
from matplotlib.ticker import MaxNLocator

import matplotlib

from Minimizer import argmin_H
from data_generation import generate_data
from minimizer_helper import generate_phi

matplotlib.use('TkAgg')
import os


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


@dataclass
class Parameters:
    nf: int = 2
    n: int = 150
    m: int = 75
    K = [1, 1]  # We use a single type of basis in each frame
    type_basis = [["gaussian"], ["gaussian"]]  # We use a single Gaussian basis for each frame
    K_st = [0, 1, 2]  # Just to get the indexing access of the array right
    sigma: float = 1.0

    x: np.ndarray = None
    t: np.ndarray = None
    t_start: float = -10.0
    t_end: float = 10.0


    degree = [3, 4]  # We use a quadratic polynomial and a cubic polynomial for both the frames respectively
    degree_st = [0, 3, 7]  # Just to get the indexing access of the array right
    beta_init = [[0.01, -0, -1.0], [0.001, -0, -0.0, 1.0]]  # Initial guess value for the coefficients of the shifts
    type_shift = ["polynomial", "polynomial"]  # We use polynomial shifts for both the frames
    beta = [[1.0, -1.0, -1.0], [0.07, 0.2, -2.5, 0.01]]
    center_of_matrix = [20, 70]

    alpha_solver_ck: float = 100000
    alpha_solver_lamda_1: float = 0.1

    beta_solver_tau: float = 0.0000001
    beta_solver_sigma: float = 0.0000001  # 0.99 / beta_solver_tau
    beta_solver_lamda_2: float = 0.1
    beta_solver_rho_n: float = 1.0
    beta_solver_gtol: float = 1e-3
    beta_solver_maxit: int = 5

    gtol: float = 1e-8
    maxit: int = 1000000


if __name__ == '__main__':

    # Instantiate the constants for the optimization
    param = Parameters()
    print(param)

    # Generate the data
    Q, _, _ = generate_data(param, param.beta)

    # Optimize over alpha and beta
    alpha, beta, J, R = argmin_H(Q, param)

    # Reconstruct the individual frames after separation and convergence
    phiBeta = generate_phi(param, beta)
    Q1 = np.einsum('ijk,kj->ij', phiBeta[0], alpha[param.K_st[0]:param.K_st[1], :])
    Q2 = np.einsum('ijk,kj->ij', phiBeta[1], alpha[param.K_st[1]:param.K_st[2], :])


    # Plots the results
    impath = "plots/QuadCubic/"  # For plots
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
    axs[3].imshow(Q1+Q2, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
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