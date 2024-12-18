import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import os
from time import perf_counter

from Minimizer import argmin_H
from NsPOD import ShapeNet, NuclearNormAutograd
from data_generation import generate_data_singleframe_torch, generate_data_faded_singleframe_torch

import torch
import torch.optim as optim

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

device_global = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cpu = torch.device("cpu")
gpu = torch.device("mps")
# seed = 5
# torch.manual_seed(seed)


class Parameters:
    def __init__(self, device):
        self.nf = 1
        self.n = 500
        self.m = 500
        self.sigma = torch.tensor(4.0, dtype=torch.float32)  # Gaussian variance for each frame

        self.t_start = -10.0
        self.t_end = 10.0
        self.x = torch.arange(0, self.n, dtype=torch.float32)
        self.t = torch.linspace(self.t_start, self.t_end, self.m, dtype=torch.float32)

        self.degree = torch.tensor(2, dtype=torch.int32)  # We use a linear polynomial (linear shift)
        self.beta = torch.tensor([-8.5, 2], dtype=torch.float32)
        self.center_of_matrix = torch.tensor(250, dtype=torch.float32)

        self.alpha_solver_ck = torch.tensor(10000, dtype=torch.float32)
        self.alpha_solver_lamda_1 = torch.tensor(0.1, dtype=torch.float32)

        self.beta_solver_tau = 0.005
        self.beta_solver_sigma = torch.tensor(0.005, dtype=torch.float32)  # 0.99 / beta_solver_tau
        self.beta_solver_lamda_2 = torch.tensor(0.1, dtype=torch.float32)
        self.beta_solver_rho_n = 1.0
        self.beta_solver_gtol = 1e-3
        self.beta_solver_maxit = 5

        self.gtol = 1e-3
        self.device = device

        self.num_epochs = 1000
        self.lamda_k = 0.1

    def to_device(self):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(self.device))


if __name__ == '__main__':

    # Instantiate the constants for the optimization
    params = Parameters(cpu)
    params.to_device()

    # Generate the data
    Q = generate_data_faded_singleframe_torch(params)
    Q = Q.to(params.device)

    # Initialize the guess values for the coefficients of the shifts
    beta = torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=True, device=params.device)
    # Initialize alpha (alpha => R^{(K[1] + K[2] + ......) X m})
    alpha = torch.zeros(params.m, dtype=torch.float32, requires_grad=True, device=params.device)

    # Instantiate the model
    model = ShapeNet(params.n, params.m, params.x, params.t)
    jit_model = torch.jit.script(model)
    jit_model.to(params.device)

    # Instantiate the optimizer
    optimizer = optim.AdamW(jit_model.parameters(), lr=0.005)

    # Epoch loop for training
    for epoch in range(params.num_epochs + 1):
        print("\n**************************************************************")
        optimizer.zero_grad()

        # Forward pass
        f1 = jit_model(beta)  # This gives us the original and stationary frame
        f1_jac = jit_model.Jacobian_forward(beta).view(params.n, params.m, 2)  # This gives the Jacobian of the Shape function

        # Losses
        frobenius_loss = torch.norm(Q - torch.einsum('ij,j->ij', f1.view(params.n, params.m), alpha), 'fro') ** 2

        # Constraints
        linearity = 

        # Backward pass
        frobenius_loss.backward(retain_graph=True)

        # Stepping the optimizer
        optimizer.step()

        # Decouple the coefficients from the graph for further computation
        with torch.no_grad():
            # Call the minimizer for tensor approach (Optimize over alpha and beta)
            alpha, beta = argmin_H(Q, f1.view(params.n, params.m), f1_jac, alpha, beta, params)

        print(
            f'Epoch {epoch}/{params.num_epochs}, Frob Loss: {frobenius_loss.item()}, '
            f'Coefficients: {beta}')

        # Reconnect with the computational graph
        beta.requires_grad_()
        alpha.requires_grad_()

    # Reconstruct the individual frames after separation and convergence
    Q = Q.cpu().detach().numpy()
    Q_shifted = torch.einsum('ij,j->ij', f1.view(params.n, params.m), alpha).cpu().detach().numpy()
    # Q_stationary = f1_stat.view(params.n, params.m).cpu().detach().numpy()

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
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
    # Original
    im2 = axs[0].pcolormesh(Q.T, vmin=vmin, vmax=vmax, cmap='viridis')
    axs[0].set_title(r"$Q$")
    axs[0].set_ylabel(r"$t$")
    axs[0].set_xlabel(r"$x$")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='10%', pad=0.08)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    # Frame shifted
    im3 = axs[1].pcolormesh(Q_shifted.T, vmin=vmin, vmax=vmax, cmap='viridis')
    axs[1].set_title(r"$\mathcal{T}^1Q^1$")
    axs[1].set_ylabel(r"$t$")
    axs[1].set_xlabel(r"$x$")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='10%', pad=0.08)
    fig.colorbar(im3, cax=cax, orientation='vertical')

    # Frame stationary
    # im4 = axs[2].pcolormesh(Q_stationary.T, vmin=vmin, vmax=vmax, cmap='viridis')
    # axs[2].set_title(r"$Q^1$")
    # axs[2].set_ylabel(r"$t$")
    # axs[2].set_xlabel(r"$x$")
    # axs[2].set_xticks([])
    # axs[2].set_yticks([])
    # divider = make_axes_locatable(axs[2])
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im4, cax=cax, orientation='vertical')

    fig.savefig(impath + "Q_opti", dpi=300, transparent=True)