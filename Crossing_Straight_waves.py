import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import shift
from math import sqrt
from scipy.sparse import diags
from scipy.linalg import cholesky
from dataclasses import dataclass
from matplotlib.ticker import MaxNLocator
import os

import matplotlib

from Minimizer import argmin_H
from data_generation import generate_data
from minimizer_helper import generate_phi

matplotlib.use('TkAgg')

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


impath = "plots/Straight/"  # For plots
os.makedirs(impath, exist_ok=True)


@dataclass
class Parameters:
    nf: int = 2
    n: int = 300
    m: int = 150
    K = [1, 1]  # We use a single type of basis in each frame
    type_basis = [["gaussian"], ["gaussian"]]  # We use a single Gaussian basis for each frame
    K_st = [0, 1, 2]  # Just to get the indexing access of the array right
    sigma = [[4.0], [4.0]]  # Gaussian variance for each frame

    x: np.ndarray = None
    t: np.ndarray = None
    t_start: float = -10.0
    t_end: float = 10.0

    degree = [2, 2]  # We use a linear polynomial for both the frames
    degree_st = [0, 2, 4]  # Just to get the indexing access of the array right
    beta_init = [[2, -1], [-2, 1]]  # Initial guess value for the coefficients of the shifts
    type_shift = ["polynomial", "polynomial"]  # We use polynomial shifts for both the frames
    beta = [[-10, 1.5], [8, -2.0]]
    center_of_matrix = [150, 150]

    alpha_solver_ck: float = 10000
    alpha_solver_lamda_1: float = 0.1

    beta_solver_tau: float = 0.00005
    beta_solver_sigma: float = 0.00005   # 0.99 / beta_solver_tau
    beta_solver_lamda_2: float = 0.1
    beta_solver_rho_n: float = 1.0
    beta_solver_gtol: float = 1e-3
    beta_solver_maxit: int = 5

    gtol: float = 1e-10
    maxit: int = 50000


if __name__ == '__main__':

    # Instantiate the constants for the optimization
    param = Parameters()
    print(param)

    # Generate the data
    Q, Q1, Q2 = generate_data(param, param.beta)

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

    # # Plot the data
    # fig1 = plt.figure(figsize=(10, 5))
    # ax1 = fig1.add_subplot(121)
    # vmin = np.min(Q)
    # vmax = np.max(Q)
    # im1 = ax1.pcolormesh(Q1.T, vmin=vmin, vmax=vmax, cmap='viridis')
    # ax1.set_xlabel(r"$x$")
    # ax1.set_ylabel(r"$t$")
    # ax1.axis('off')
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig1.colorbar(im1, cax=cax, orientation='vertical')
    #
    # ax2 = fig1.add_subplot(122)
    # im2 = ax2.pcolormesh(Q2.T, vmin=vmin, vmax=vmax, cmap='viridis')
    # ax2.set_xlabel(r"$x$")
    # ax2.set_ylabel(r"$t$")
    # ax2.axis('off')
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig1.colorbar(im2, cax=cax, orientation='vertical')
    # fig1.supylabel(r"time $t$")
    # fig1.supxlabel(r"space $x$")
    # fig1.savefig(impath + "Q_sep", dpi=300, transparent=True)




    exit()

    # Optimize over alpha and beta
    alpha, beta, J, R = argmin_H(Q, param)

    # Reconstruct the individual frames after separation and convergence
    phiBeta = generate_phi(param, beta)
    Q1 = np.einsum('ijk,kj->ij', phiBeta[0], alpha[param.K_st[0]:param.K_st[1], :])
    Q2 = np.einsum('ijk,kj->ij', phiBeta[1], alpha[param.K_st[1]:param.K_st[2], :])

    # Plots the results
    # Plot the separated frames
    fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharey=True, sharex=True)
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

    # Reconstructed
    im4 = axs[3].pcolormesh((Q1 + Q2).T, vmin=vmin, vmax=vmax, cmap='viridis')
    axs[3].set_title(r"$\tilde{Q}$")
    axs[3].set_ylabel(r"$t$")
    axs[3].set_xlabel(r"$x$")
    axs[3].set_xticks([])
    axs[3].set_yticks([])

    plt.colorbar(im4, ax=axs.ravel().tolist(), orientation='vertical')

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