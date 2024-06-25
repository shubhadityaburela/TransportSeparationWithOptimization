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
    nf: int = 3
    n: int = None
    m: int = None
    sigma = [[1.0, 6.0], [3.0], [1.0, 6.0]]  # Gaussian variance for each frame
    K = [2, 1, 2]  # We use a single type of basis in each frame
    type_basis = [["gaussian", "gaussian"], ["gaussian"], ["gaussian", "gaussian"]]  # We use a single Gaussian basis for each frame
    K_st = [0, 2, 3, 5]  # Just to get the indexing access of the array right

    x: np.ndarray = None
    t: np.ndarray = None

    degree = [2, 1, 2]  # We use a linear polynomial for the first and last frame and a constant for the middle frame
    degree_st = [0, 2, 3, 5]  # Just to get the indexing access of the array right
    beta_init = [[0.01, -0.5], [0], [-0.08, 1.0]]  # Initial guess value for the coefficients of the shifts
    type_shift = ["polynomial", "polynomial", "polynomial"]  # We use polynomial shifts for all the frames
    center_of_matrix = [0, 0, 0]

    alpha_solver_ck: float = 50000
    alpha_solver_lamda_1: float = 10

    beta_solver_tau: float = 0.00005
    beta_solver_sigma: float = 0.00005
    beta_solver_lamda_2: float = 0.1
    beta_solver_rho_n: float = 1.0
    beta_solver_gtol: float = 1e-3
    beta_solver_maxit: int = 5

    gtol: float = 1e-8
    maxit: int = 100

    type: str = 'Wildlandfire1D'


if __name__ == '__main__':

    # Load the wild land fire data
    q = np.load('data/Wildlandfire_1d/SnapShotMatrix558.49.npy', allow_pickle=True)
    X = np.load('data/Wildlandfire_1d/1D_Grid.npy', allow_pickle=True)
    t = np.load('data/Wildlandfire_1d/Time.npy', allow_pickle=True)
    x = X[0]
    Nx = int(np.size(x))
    Nt = int(np.size(t))
    Q = q[:Nx, :]

    # Normalize the input data
    Q = (Q - Q.min())/(Q.max() - Q.min())

    # Instantiate the constants for the optimization
    param = Parameters()
    param.n = Nx
    param.m = Nt
    param.x = x
    param.t = t
    param.center_of_matrix = [x[-1] // 2, x[-1] // 2, x[-1] // 2]

    # Optimize over alpha and beta
    alpha, beta, J, R = argmin_H(Q, param)

    # Reconstruct the individual frames after separation and convergence
    phiBeta = generate_phi(param, beta)
    Q1 = np.einsum('ijk,kj->ij', phiBeta[0], alpha[param.K_st[0]:param.K_st[1], :])
    Q2 = np.einsum('ijk,kj->ij', phiBeta[1], alpha[param.K_st[1]:param.K_st[2], :])
    Q3 = np.einsum('ijk,kj->ij', phiBeta[2], alpha[param.K_st[2]:param.K_st[3], :])


    # Plots the results
    impath = "plots/Wildlandfire1D/"  # For plots
    os.makedirs(impath, exist_ok=True)

    # Plot the separated frames
    fig, axs = plt.subplots(1, 5, figsize=(20, 6))
    vmin = np.min(Q)
    vmax = np.max(Q)
    # Original
    axs[0].imshow(Q, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
    # axs[0].imshow(Q, cmap='hot', aspect='auto')
    axs[0].set_title('Original')
    axs[0].set_xlabel('$t_j$')
    axs[0].set_ylabel('$x_i$')
    # Frame 1
    axs[1].imshow(Q1, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
    # axs[1].imshow(Q1, cmap='hot', aspect='auto')
    axs[1].set_title('Frame 1')
    axs[1].set_xlabel('$t_j$')
    axs[1].set_ylabel('$x_i$')
    # Frame 2
    axs[2].imshow(Q2, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
    # axs[2].imshow(Q2, cmap='hot', aspect='auto')
    axs[2].set_title('Frame 2')
    axs[2].set_xlabel('$t_j$')
    axs[2].set_ylabel('$x_i$')
    # Frame 3
    axs[3].imshow(Q3, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
    # axs[3].imshow(Q3, cmap='hot', aspect='auto')
    axs[3].set_title('Frame 3')
    axs[3].set_xlabel('$t_j$')
    axs[3].set_ylabel('$x_i$')
    # Reconstructed
    axs[4].imshow(Q1+Q2+Q3, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
    # axs[4].imshow(Q1 + Q2 + Q3, cmap='hot', aspect='auto')
    axs[4].set_title('Reconstructed')
    axs[4].set_xlabel('$t_j$')
    axs[4].set_ylabel('$x_i$')
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
