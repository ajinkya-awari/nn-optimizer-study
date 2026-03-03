"""
loss_landscape.py — 2D loss landscape visualisation using filter-normalised
random directions (Li et al., 2018, "Visualizing the Loss Landscape of
Neural Nets", NeurIPS 2018).

Two directions d1, d2 are sampled and filter-normalised to account for
scale invariance of ReLU networks. The loss is then evaluated on a grid
of perturbations θ* + α*d1 + β*d2, where θ* is the trained model's weights.

This is computationally expensive — the grid is 25x25 by default (625 model
evaluations). Bump GRID_SIZE down to 15 if it's too slow.

Outputs : results/loss_landscape.png
          results/loss_landscape_3d.png
Usage   : python loss_landscape.py --optimizer adam
          (loads checkpoints/Adam_final.pt by default)
"""

import os
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


from model import get_model


# ── Configuration ─────────────────────────────────────────────────────────────

GRID_SIZE    = 25      # reduce to 15 if this takes too long
RANGE        = 1.0     # perturbation range in each direction, ±RANGE
BATCH_LIMIT  = 5       # how many batches to use per grid point — good tradeoff
RESULTS_DIR  = "results"
CKPT_DIR     = "checkpoints"
SEED         = 42


# ── Filter normalisation ──────────────────────────────────────────────────────

def filter_normalise(direction: list[torch.Tensor], reference: list[torch.Tensor]):
    """
    Scale each filter in 'direction' to match the norm of the corresponding
    filter in 'reference'. This is the key step from Li et al. 2018 —
    without it, the landscape would be dominated by large-norm parameters.

    For FC layers we treat the whole weight matrix as a single "filter".
    """
    for d_param, ref_param in zip(direction, reference):
        if d_param.dim() > 1:
            # convolutional or FC weight: normalise filter-by-filter
            for f_idx in range(d_param.size(0)):
                ref_norm = ref_param[f_idx].norm()
                d_norm   = d_param[f_idx].norm()
                if d_norm > 1e-10:
                    d_param[f_idx].mul_(ref_norm / d_norm)
        else:
            # bias or batchnorm param — scalar normalisation
            ref_norm = ref_param.norm()
            d_norm   = d_param.norm()
            if d_norm > 1e-10:
                d_param.mul_(ref_norm / d_norm)


def sample_random_direction(reference_params: list[torch.Tensor]) -> list[torch.Tensor]:
    """Gaussian random direction, same shape as the model parameters."""
    direction = [torch.randn_like(p) for p in reference_params]
    filter_normalise(direction, reference_params)
    return direction


# ── Loss evaluation on perturbed weights ─────────────────────────────────────

@torch.no_grad()
def eval_loss_at_perturbation(
    model        : nn.Module,
    base_params  : list[torch.Tensor],
    d1           : list[torch.Tensor],
    d2           : list[torch.Tensor],
    alpha        : float,
    beta         : float,
    loader       : DataLoader,
    criterion    : nn.Module,
    device       : str,
    batch_limit  : int = BATCH_LIMIT,
) -> float:
    """
    θ_perturbed = θ* + α*d1 + β*d2
    Evaluate cross-entropy loss at this perturbed point.
    """
    # temporarily overwrite model weights
    params = list(model.parameters())
    for p, base, dir1, dir2 in zip(params, base_params, d1, d2):
        p.copy_(base + alpha * dir1 + beta * dir2)

    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        out = model(batch_x)
        loss = criterion(out, batch_y).item()
        total_loss += loss
        n_batches  += 1
        if n_batches >= batch_limit:
            break

    return total_loss / n_batches


# ── Data loader (small subset for speed) ─────────────────────────────────────

def get_small_loader(dataset: str = "cifar10", n_samples: int = 1024) -> DataLoader:
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    indices = list(range(min(n_samples, len(ds))))
    return DataLoader(Subset(ds, indices), batch_size=128, shuffle=False, num_workers=0)


# ── Main landscape sweep ──────────────────────────────────────────────────────

def compute_landscape(
    checkpoint_path : str,
    dataset         : str  = "cifar10",
    grid_size       : int  = GRID_SIZE,
    perturbation_range: float = RANGE,
    device          : str  = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (alphas, betas, loss_grid) where loss_grid[i,j] is the loss
    at perturbation (alphas[i], betas[j]).
    """
    torch.manual_seed(SEED)

    model = get_model(dataset, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    # store a frozen copy of the trained weights
    base_params = [p.detach().clone() for p in model.parameters()]

    loader    = get_small_loader(dataset)
    criterion = nn.CrossEntropyLoss()

    d1 = sample_random_direction(base_params)
    d2 = sample_random_direction(base_params)

    coords = np.linspace(-perturbation_range, perturbation_range, grid_size)
    loss_grid = np.zeros((grid_size, grid_size))

    print(f"Computing loss landscape ({grid_size}x{grid_size} = {grid_size**2} evaluations)...")

    for i, alpha in enumerate(coords):
        for j, beta in enumerate(coords):
            loss_grid[i, j] = eval_loss_at_perturbation(
                model, base_params, d1, d2,
                alpha, beta, loader, criterion, device,
            )
        if (i + 1) % 5 == 0:
            print(f"  Row {i+1}/{grid_size} done")

    # restore original weights before returning
    params = list(model.parameters())
    for p, base in zip(params, base_params):
        p.copy_(base)

    return coords, coords, loss_grid


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_landscape_2d(alphas, betas, loss_grid, out_dir: str, label: str = ""):
    fig, ax = plt.subplots(figsize=(7, 6))

    A, B = np.meshgrid(alphas, betas)

    # log scale makes the basin shape much more visible
    log_loss = np.log(loss_grid + 1e-8)

    contour_fill = ax.contourf(A, B, log_loss.T, levels=40, cmap="RdYlBu_r")
    ax.contour(A, B, log_loss.T, levels=15, colors="white", linewidths=0.4, alpha=0.5)

    plt.colorbar(contour_fill, ax=ax, label="log(Loss)")
    ax.scatter([0], [0], color="white", s=80, zorder=10, marker="*", label=f"{label} minimum")
    ax.set_xlabel(r"Direction $d_1$  (filter-normalised)")
    ax.set_ylabel(r"Direction $d_2$  (filter-normalised)")
    ax.set_title(f"Loss Landscape — {label}\n(Li et al. 2018 filter normalisation)")
    ax.legend(loc="lower right", fontsize=8)

    path = os.path.join(out_dir, "loss_landscape.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_landscape_3d(alphas, betas, loss_grid, out_dir: str, label: str = ""):
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    A, B = np.meshgrid(alphas, betas)
    log_loss = np.log(loss_grid.T + 1e-8)

    surf = ax.plot_surface(A, B, log_loss, cmap=cm.RdYlBu_r, linewidth=0, antialiased=True, alpha=0.85)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="log(Loss)")

    ax.set_xlabel(r"$d_1$")
    ax.set_ylabel(r"$d_2$")
    ax.set_zlabel("log(Loss)")
    ax.set_title(f"3D Loss Surface — {label}")
    ax.view_init(elev=30, azim=-60)

    path = os.path.join(out_dir, "loss_landscape_3d.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_landscape(optimizer_name: str = "Adam", dataset: str = "cifar10"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # map optimizer name to checkpoint filename
    ckpt_name = optimizer_name.replace("+", "_") + "_final.pt"
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}\n"
            "Run train.py first, or pass --optimizer with an available checkpoint name."
        )

    print(f"Loading checkpoint: {ckpt_path}")
    alphas, betas, grid = compute_landscape(ckpt_path, dataset, device=device)

    plot_landscape_2d(alphas, betas, grid, RESULTS_DIR, label=optimizer_name)
    plot_landscape_3d(alphas, betas, grid, RESULTS_DIR, label=optimizer_name)

    print("Loss landscape done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="Adam",
                        help="Optimizer name matching the checkpoint (default: Adam)")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    args = parser.parse_args()
    run_landscape(args.optimizer, args.dataset)
