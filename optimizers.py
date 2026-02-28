"""
optimizers.py — Hand-rolled SGD and SGD-with-Momentum implementations.

Written against PyTorch's optimizer interface so they slot into the same
training loop as Adam/RMSprop without any special casing. The math is
deliberately kept transparent — each update step maps directly to the
equations in the docstrings.

Outputs : used by train.py
Usage   : imported as a module
"""

import torch
from torch.optim import Optimizer


# ── Vanilla SGD ──────────────────────────────────────────────────────────────

class SGDCustom(Optimizer):
    """
    Stochastic Gradient Descent, implemented from first principles.

    Update rule:
        θ_{t+1} = θ_t − η · ∇_θ L(θ_t)

    where η is the learning rate and ∇_θ L is the gradient of the loss
    w.r.t. parameters θ at step t. No weight decay, no bells and whistles —
    just the bare update to make the optimizer differences as legible as possible.
    """

    def __init__(self, params, lr: float = 0.01):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eta = group["lr"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                # θ_{t+1} = θ_t - η * ∇L
                # in-place subtract keeps memory usage down
                grad = param.grad
                param.add_(grad, alpha=-eta)

        return loss


# ── SGD with Classical Momentum ──────────────────────────────────────────────

class SGDMomentum(Optimizer):
    """
    SGD with classical (heavy ball) momentum.

    Update rule:
        v_{t+1} = β · v_t + ∇_θ L(θ_t)
        θ_{t+1} = θ_t − η · v_{t+1}

    β is the momentum coefficient (typically 0.9). The velocity accumulates
    gradient history, dampening oscillations in high-curvature directions.
    See Polyak (1964) for the original formulation; Sutton (1986) for the
    stochastic variant used here.

    Note: this is classical momentum, not Nesterov. Nesterov evaluates the
    gradient at the "lookahead" position — worth trying but adds complexity
    that obscures the comparison here.
    TODO: add a nesterov=True flag and compare convergence
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.9):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")

        defaults = {"lr": lr, "momentum": momentum}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eta  = group["lr"]
            beta = group["momentum"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # velocity lives in optimizer state — initialised to zero on first call
                state = self.state[param]
                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(param)

                v = state["velocity"]

                # v_{t+1} = β * v_t + ∇L  (accumulate gradient into velocity)
                v.mul_(beta).add_(grad)

                # θ_{t+1} = θ_t - η * v_{t+1}
                param.add_(v, alpha=-eta)

        return loss


# ── Gradient norm helper ─────────────────────────────────────────────────────

def compute_gradient_norm(model) -> float:
    """
    ||∇L||_2 across all parameters. Useful for spotting vanishing/exploding
    gradients and for comparing how aggressively different optimizers
    move through the loss landscape.
    """
    total_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm_sq += param.grad.detach().norm(2).item() ** 2
    return total_norm_sq ** 0.5


def compute_gradient_stats(model) -> dict:
    """
    Returns norm, mean absolute value, and an approximate rank estimate
    of the full gradient vector.

    The rank estimate (fraction of singular values above 1% of max) is
    a rough proxy for how compressible the gradient is — relevant to
    low-rank compression methods like PowerSGD. Not cheap to compute,
    so only call this every N epochs, not every batch.
    """
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().view(-1))

    if not grads:
        return {"norm": 0.0, "mean_abs": 0.0, "sparsity": 0.0}

    flat = torch.cat(grads)
    norm     = flat.norm(2).item()
    mean_abs = flat.abs().mean().item()
    # sparsity: fraction of gradient elements below 1e-4 in absolute value
    # high sparsity = gradient compresses well; low sparsity = dense update
    sparsity = (flat.abs() < 1e-4).float().mean().item()

    return {"norm": norm, "mean_abs": mean_abs, "sparsity": sparsity}
