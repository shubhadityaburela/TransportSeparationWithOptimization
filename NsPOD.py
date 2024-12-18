import torch
import torch.nn as nn


# ============================================================================ #
#                    Neural network class and functions                        #
# ============================================================================ #
@torch.jit.script
def nuclear_norm(input_matrix: torch.Tensor) -> torch.Tensor:
    # Forward computation
    return torch.linalg.matrix_norm(input_matrix, ord="nuc")


@torch.jit.script
def nuclear_norm_grad(input_matrix: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    u, s, v = torch.linalg.svd(input_matrix, full_matrices=False)
    rank = (s > 0).sum().item()
    eye_approx = torch.diag_embed((s > 0).to(input_matrix.dtype)[:rank])
    grad_input = torch.matmul(u[:, :rank], eye_approx)
    grad_input = torch.matmul(grad_input, v[:, :rank].transpose(-2, -1))
    return grad_input * grad_output.unsqueeze(-1).unsqueeze(-1)


class NuclearNormAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_matrix):
        ctx.save_for_backward(input_matrix)
        return nuclear_norm(input_matrix)

    @staticmethod
    def backward(ctx, grad_output):
        input_matrix, = ctx.saved_tensors
        return nuclear_norm_grad(input_matrix, grad_output)


class ShapeNet(nn.Module):
    def __init__(self, n, m, x, t):
        super(ShapeNet, self).__init__()

        self.elu = nn.ELU()
        self.n = n
        self.m = m
        self.x = x
        self.t = t

        # Subnetwork for f^1
        self.f1_fc1 = nn.Linear(2, 15)
        self.f1_fc2 = nn.Linear(15, 50)
        self.f1_fc3 = nn.Linear(50, 100)
        self.f1_fc4 = nn.Linear(100, self.n * self.m)

    def forward(self, coeffs):
        # Pathway for f^1 and shift^1
        f1 = self.elu(self.f1_fc1(coeffs))
        f1 = self.elu(self.f1_fc2(f1))
        f1 = self.elu(self.f1_fc3(f1))
        f1 = self.f1_fc4(f1)

        return f1

    @torch.jit.ignore
    def Jacobian_forward(self, coeffs):
        f1_jac = torch.func.jacfwd(self.forward)(coeffs)
        return f1_jac

    # @torch.jit.export
    # def wrapper(self, coeffs):
    #     return self.forward(coeffs)
