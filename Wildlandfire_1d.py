import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


# Plots the results
impath = "plots/Wildlandfire1D/"  # For plots
os.makedirs(impath, exist_ok=True)


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
    sigma = [[3.0], [3.0], [3.0]]  # Gaussian variance for each frame
    K = [1, 1, 1]  # We use a single type of basis in each frame
    type_basis = [["gaussian"], ["gaussian"], ["gaussian"]]  # We use a single Gaussian basis for each frame
    K_st = [0, 1, 2, 3]  # Just to get the indexing access of the array right

    x: np.ndarray = None
    t: np.ndarray = None

    degree = [2, 1, 2]  # We use a linear polynomial for the first and last frame and a constant for the middle frame
    degree_st = [0, 2, 3, 5]  # Just to get the indexing access of the array right
    beta_init = [[0.01, -0.5], [0], [-0.08, 1.0]]  # Initial guess value for the coefficients of the shifts
    type_shift = ["polynomial", "polynomial", "polynomial"]  # We use polynomial shifts for all the frames
    center_of_matrix = [0, 0, 0]

    alpha_solver_ck: float = 5000000
    alpha_solver_lamda_1: float = 0.1

    beta_solver_tau: float = 0.0000005
    beta_solver_sigma: float = 0.0000005
    beta_solver_lamda_2: float = 0.1
    beta_solver_rho_n: float = 1.0
    beta_solver_gtol: float = 1e-3
    beta_solver_maxit: int = 5

    gtol: float = 1e-8
    maxit: int = 100000

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

    # Plot the data
    fig1 = plt.figure(figsize=(5, 5))
    ax1 = fig1.add_subplot(111)
    vmin = np.min(Q)
    vmax = np.max(Q)
    im1 = ax1.pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap='viridis')
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$t$")
    ax1.axis('off')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='10%', pad=0.08)
    fig1.colorbar(im1, cax=cax, orientation='vertical')
    fig1.supylabel(r"time $t$")
    fig1.supxlabel(r"space $x$")
    fig1.savefig(impath + "Q", dpi=300, transparent=True)

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

    # Plot the separated frames
    fig, axs = plt.subplots(1, 5, figsize=(20, 6), sharey=True, sharex=True)
    # Original
    axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap='viridis')
    axs[0].set_title(r"$Q$")
    axs[0].set_ylabel(r"$t$")
    axs[0].set_xlabel(r"$x$")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Frame 1
    axs[1].pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap='viridis')
    axs[1].set_title(r"$\mathcal{T}^1Q^1$")
    axs[1].set_ylabel(r"$t$")
    axs[1].set_xlabel(r"$x$")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    # Frame 2
    axs[2].pcolormesh(Q2.T, vmin=vmin, vmax=vmax, cmap='viridis')
    axs[2].set_title(r"$\mathcal{T}^2Q^2$")
    axs[2].set_ylabel(r"$t$")
    axs[2].set_xlabel(r"$x$")
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    # Frame 3
    axs[3].pcolormesh(Q3.T, vmin=vmin, vmax=vmax, cmap='viridis')
    axs[3].set_title(r"$\mathcal{T}^3Q^3$")
    axs[3].set_ylabel(r"$t$")
    axs[3].set_xlabel(r"$x$")
    axs[3].set_xticks([])
    axs[3].set_yticks([])

    # Frame 4
    im5 = axs[4].pcolormesh((Q1+Q2+Q3).T, vmin=vmin, vmax=vmax, cmap='viridis')
    axs[4].set_title(r"$\tilde{Q}$")
    axs[4].set_ylabel(r"$t$")
    axs[4].set_xlabel(r"$x$")
    axs[4].set_xticks([])
    axs[4].set_yticks([])

    plt.colorbar(im5, ax=axs.ravel().tolist(), orientation='vertical')
    fig.savefig(impath + "Q_opti", dpi=300, transparent=True)

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
