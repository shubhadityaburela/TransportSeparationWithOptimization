import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataclasses import dataclass
from matplotlib.ticker import MaxNLocator
import os

import matplotlib

from Minimizer import argmin_H
from data_generation import generate_data, generate_data_faded, generate_data_faded_singleframe, \
    generate_data_singleframe
from minimizer_helper import generate_phi
from numba import float64, int64
from numba.experimental import jitclass

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

spec = [
    ('nf', int64),
    ('n', int64),
    ('m', int64),
    ('K', int64[:]),
    ('type_basis', int64[:]),
    ('K_st', int64[:]),
    ('sigma', float64[:]),
    ('t_start', float64),
    ('t_end', float64),
    ('x', float64[:]),
    ('t', float64[:]),
    ('degree', int64[:]),
    ('degree_st', int64[:]),
    ('beta_init', float64[:]),
    ('type_shift', int64[:]),
    ('beta', float64[:]),
    ('center_of_matrix', float64[:]),
    ('alpha_solver_ck', float64),
    ('alpha_solver_lamda_1', float64),
    ('beta_solver_tau', float64),
    ('beta_solver_sigma', float64),
    ('beta_solver_lamda_2', float64),
    ('beta_solver_rho_n', float64),
    ('beta_solver_gtol', float64),
    ('beta_solver_maxit', int64),
    ('gtol', float64),
    ('maxit', int64)
]


@jitclass(spec)
class Parameters:
    def __init__(self):
        self.nf = 1
        self.n = 500
        self.m = 500
        self.K = np.array([1], dtype=np.int64)  # We use a single type of basis in each frame
        self.type_basis = np.array([1], dtype=np.int64)  # We use a single Gaussian basis for each frame  # 1 is for Gaussian basis
        self.K_st = np.array([0, 1], dtype=np.int64)  # Just to get the indexing access of the array right
        self.sigma = np.array([4.0], dtype=np.float64)  # Gaussian variance for each frame

        self.t_start = -10.0
        self.t_end = 10.0
        self.x = np.arange(0, self.n, dtype=np.float64)
        self.t = np.linspace(self.t_start, self.t_end, self.m)

        self.degree = np.array([2], dtype=np.int64)  # We use a linear polynomial for both the frames
        self.degree_st = np.array([0, 2], dtype=np.int64)  # Just to get the indexing access of the array right
        self.beta_init = np.array([2.0, -1.0], dtype=np.float64)  # Initial guess value for the coefficients of the shifts
        self.type_shift = np.array([1], dtype=np.int64)  # We use polynomial shifts for both the frames, Code is 1 for them
        self.beta = np.array([-10.0, 1.5], dtype=np.float64)
        self.center_of_matrix = np.array([250], dtype=np.float64)

        self.alpha_solver_ck = 10000
        self.alpha_solver_lamda_1 = 0.1

        self.beta_solver_tau = 0.0005
        self.beta_solver_sigma = 0.0005  # 0.99 / beta_solver_tau
        self.beta_solver_lamda_2 = 0.1
        self.beta_solver_rho_n = 1.0
        self.beta_solver_gtol = 1e-3
        self.beta_solver_maxit = 5

        self.gtol = 1e-10
        self.maxit = 10000


if __name__ == '__main__':

    # Instantiate the constants for the optimization
    param = Parameters()

    # Generate the data
    Q = generate_data_faded_singleframe(param, param.beta)

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

    # Optimize over alpha and beta
    alpha, beta, J, R = argmin_H(Q, param)

    # Reconstruct the individual frames after separation and convergence
    phiBeta = generate_phi(param, beta)
    Q_shifted = np.einsum('ij,kj->ij', phiBeta[0], alpha)

    # Plots the results
    # Plot the separated frames
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True, sharex=True)
    # Original
    axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap='viridis')
    axs[0].set_title(r"$Q$")
    axs[0].set_ylabel(r"$t$")
    axs[0].set_xlabel(r"$x$")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Frame 1
    axs[1].pcolormesh(Q_shifted.T, vmin=vmin, vmax=vmax, cmap='viridis')
    axs[1].set_title(r"$\mathcal{T}^1Q^1$")
    axs[1].set_ylabel(r"$t$")
    axs[1].set_xlabel(r"$x$")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

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