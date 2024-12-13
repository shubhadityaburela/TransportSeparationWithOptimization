import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataclasses import dataclass
from matplotlib.ticker import MaxNLocator
import os

from Minimizer import argmin_H
from data_generation import generate_data, generate_data_faded, generate_data_faded_singleframe, \
    generate_data_singleframe, generate_data_singleframe_torch
from minimizer_helper import generate_phi
from numba import float64, int64
from numba.experimental import jitclass

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

device_global = "cpu"   # torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Parameters:
    def __init__(self, device):
        self.nf = 1
        self.n = 1000
        self.m = 1000
        # self.K = torch.tensor([1], dtype=torch.int32)  # We use a single type of basis in each frame
        # self.type_basis = torch.tensor([1], dtype=torch.int32)  # We use a single Gaussian basis for each frame  # 1 is for Gaussian basis
        # self.K_st = torch.tensor([0, 1], dtype=torch.int32)  # Just to get the indexing access of the array right
        self.sigma = torch.tensor(4.0, dtype=torch.float32)  # Gaussian variance for each frame

        self.t_start = -10.0
        self.t_end = 10.0
        self.x = torch.arange(0, self.n, dtype=torch.float32)
        self.t = torch.linspace(self.t_start, self.t_end, self.m, dtype=torch.float32)

        self.degree = torch.tensor(2, dtype=torch.int32)  # We use a linear polynomial (linear shift)
        # self.degree_st = torch.tensor([0, 2], dtype=torch.int32)  # Just to get the indexing access of the array right
        self.beta_init = torch.tensor([2.0, -1.0], dtype=torch.float32)  # Initial guess value for the coefficients of the shifts
        # self.type_shift = torch.tensor([1], dtype=torch.int32)  # We use polynomial shifts, Code is 1 for them
        self.beta = torch.tensor([-10.0, 1.5], dtype=torch.float32)
        self.center_of_matrix = torch.tensor(500, dtype=torch.float32)

        self.alpha_solver_ck = torch.tensor(10000, dtype=torch.float32)
        self.alpha_solver_lamda_1 = torch.tensor(0.1, dtype=torch.float32)

        self.beta_solver_tau = 0.00005
        self.beta_solver_sigma = torch.tensor(0.00005, dtype=torch.float32)  # 0.99 / beta_solver_tau
        self.beta_solver_lamda_2 = torch.tensor(0.1, dtype=torch.float32)
        self.beta_solver_rho_n = 1.0
        self.beta_solver_gtol = 1e-3
        self.beta_solver_maxit = 5

        self.gtol = 1e-10
        self.maxit = 10000
        self.device = device

    def to_device(self, device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))


if __name__ == '__main__':

    # Instantiate the constants for the optimization
    params = Parameters(device_global)
    params.to_device(device_global)

    # Generate the data
    Q = generate_data_singleframe_torch(params)
    Q = Q.to(device_global)

    # Optimize over alpha and beta
    alpha, beta, J, R = argmin_H(Q, params)

    # Reconstruct the individual frames after separation and convergence
    phiBeta = generate_phi(params, beta)
    Q_shifted = torch.einsum('ij,j->ij', phiBeta, alpha)

    # Plots the results
    Q = Q.cpu().detach().numpy()
    Q_shifted = Q_shifted.cpu().detach().numpy()
    J = [tensor.cpu() for tensor in J]

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