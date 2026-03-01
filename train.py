"""
train.py — Training loop that runs the same model against all six optimizers.

Tracks per-epoch: loss, validation accuracy, gradient norm, wall-clock time,
and gradient evaluation count. Results are returned as a dict of lists and
also saved to results/raw_metrics.json so analysis.py can read them without
re-running training.

Outputs : results/raw_metrics.json, checkpoints/<optimizer>_final.pt
Usage   : python train.py  (or called from main.py)
"""

import os
import json
import time
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from model import get_model
from optimizers import SGDCustom, SGDMomentum, compute_gradient_norm, compute_gradient_stats


# ── Reproducibility ──────────────────────────────────────────────────────────

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Data loading ─────────────────────────────────────────────────────────────

def get_dataloaders(dataset: str = "cifar10", batch_size: int = 128, val_fraction: float = 0.1):
    """
    Standard CIFAR-10 pipeline with per-channel normalisation.
    We carve 10% off training for validation — gives a stable signal
    without touching the held-out test set.
    """
    if dataset == "cifar10":
        # mean/std from the CIFAR-10 dataset statistics
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        full_train = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )

    else:  # mnist
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        full_train = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    val_size   = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size

    # fixed generator so the split is the same for every optimizer run
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    # num_workers=0 on Windows — multiprocessing with DataLoader needs a proper
    # spawn context that doesn't exist inside scripts run directly on Windows.
    # On Linux/Mac you can bump this to 2-4 for a speed gain.
    n_workers = 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=256,         shuffle=False, num_workers=n_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,         shuffle=False, num_workers=n_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ── Optimizer factory ─────────────────────────────────────────────────────────

def build_optimizer(name: str, model_params, lr_override: float = None):
    """
    Central place to configure each optimizer. LR defaults are tuned to give
    roughly comparable convergence speeds — not to maximise each optimizer's
    individual performance, which would make comparisons meaningless.

    TODO: run a proper LR sweep and document the chosen values
    """
    # deliberately using common, well-known defaults where they exist
    configs = {
        "SGD"      : {"lr": 0.01},
        "SGD+Mom"  : {"lr": 0.01, "momentum": 0.9},
        "Adam"     : {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8},
        "RMSprop"  : {"lr": 1e-3, "alpha": 0.99, "eps": 1e-8},
        "Adagrad"  : {"lr": 0.01},
        "LBFGS"    : {"lr": 1.0, "max_iter": 20, "history_size": 10},
    }

    cfg = configs[name]
    if lr_override is not None:
        cfg = {**cfg, "lr": lr_override}

    if name == "SGD":
        return SGDCustom(model_params, lr=cfg["lr"])
    elif name == "SGD+Mom":
        return SGDMomentum(model_params, lr=cfg["lr"], momentum=cfg["momentum"])
    elif name == "Adam":
        return torch.optim.Adam(model_params, **cfg)
    elif name == "RMSprop":
        return torch.optim.RMSprop(model_params, **cfg)
    elif name == "Adagrad":
        return torch.optim.Adagrad(model_params, **cfg)
    elif name == "LBFGS":
        return torch.optim.LBFGS(model_params, **cfg)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ── Single training epoch ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, optimizer_name: str):
    """
    One pass through the training set. Returns (avg_loss, grad_norm, grad_evals).

    L-BFGS needs special treatment because it re-evaluates the loss multiple
    times per step — we track that separately for the efficiency comparison.
    """
    model.train()
    total_loss    = 0.0
    total_samples = 0
    grad_evals    = 0
    last_grad_norm = 0.0

    is_lbfgs = (optimizer_name == "LBFGS")

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        if is_lbfgs:
            # LBFGS requires a closure that recomputes the loss on each call
            # each call to the closure counts as one gradient evaluation
            def closure():
                nonlocal grad_evals
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                grad_evals += 1
                return loss

            loss = optimizer.step(closure)
            # LBFGS step() returns whatever the closure returns — make sure
            # it's a plain float before we accumulate it
            if isinstance(loss, torch.Tensor):
                loss = loss.detach()

        else:
            optimizer.zero_grad()
            output = model(batch_x)
            loss   = criterion(output, batch_y)
            loss.backward()
            last_grad_norm = compute_gradient_norm(model)
            optimizer.step()
            grad_evals += 1

        total_loss    += loss.detach().item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss, last_grad_norm, grad_evals


# ── Validation pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total   = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        preds = model(batch_x).argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total   += batch_y.size(0)
    return correct / total


# ── Full training run for one optimizer ──────────────────────────────────────

def train_optimizer(
    optimizer_name : str,
    train_loader   : DataLoader,
    val_loader     : DataLoader,
    dataset        : str  = "cifar10",
    n_epochs       : int  = 30,
    device         : str  = "cpu",
    save_dir       : str  = "checkpoints",
) -> dict:
    """
    Trains a fresh model from the same random initialisation using the given optimizer.
    Returns a dict of per-epoch metrics lists.
    """
    set_seed(SEED)
    model = get_model(dataset, device)
    optimizer = build_optimizer(optimizer_name, model.parameters())
    criterion = nn.CrossEntropyLoss()

    # ReduceLROnPlateau works fine with our custom optimizers because they follow
    # the standard param_groups interface. LBFGS does its own line search so
    # a plateau scheduler on top of it doesn't make sense.
    scheduler = None
    if optimizer_name not in ("LBFGS",):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

    metrics = {
        "optimizer"      : optimizer_name,
        "train_loss"     : [],
        "val_acc"        : [],
        "grad_norm"      : [],
        "grad_sparsity"  : [],   # fraction of near-zero gradient elements — proxy for compressibility
        "epoch_time_sec" : [],
        "grad_evals"     : [],
    }

    print(f"\n{'─'*55}")
    print(f"  Training with: {optimizer_name}")
    print(f"{'─'*55}")

    cumulative_grad_evals = 0

    for epoch in range(1, n_epochs + 1):
        t_start = time.perf_counter()

        avg_loss, grad_norm, epoch_grad_evals = train_epoch(
            model, train_loader, optimizer, criterion, device, optimizer_name
        )
        val_acc = evaluate(model, val_loader, device)

        # gradient stats on the last mini-batch of this epoch — not perfect but
        # consistent across optimizers and cheap enough to do every epoch
        grad_stats = compute_gradient_stats(model)

        elapsed = time.perf_counter() - t_start
        cumulative_grad_evals += epoch_grad_evals

        metrics["train_loss"].append(round(avg_loss, 6))
        metrics["val_acc"].append(round(val_acc, 6))
        metrics["grad_norm"].append(round(grad_norm, 6))
        metrics["grad_sparsity"].append(round(grad_stats["sparsity"], 4))
        metrics["epoch_time_sec"].append(round(elapsed, 3))
        metrics["grad_evals"].append(cumulative_grad_evals)

        if scheduler is not None:
            scheduler.step(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.3f}  "
                  f"‖∇L‖={grad_norm:.4f}  t={elapsed:.1f}s")

    # save checkpoint so loss_landscape.py can load the final weights
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{optimizer_name.replace('+', '_')}_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")

    return metrics


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all_optimizers(
    dataset    : str = "cifar10",
    n_epochs   : int = 30,
    batch_size : int = 128,
    results_dir: str = "results",
) -> List[dict]:
    """
    Runs all six optimizers sequentially. Each starts from the same seed so
    the only variable is the optimizer itself.
    """
    os.makedirs(results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(dataset, batch_size)

    optimizer_names = ["SGD", "SGD+Mom", "Adam", "RMSprop", "Adagrad", "LBFGS"]
    all_metrics = []

    for opt_name in optimizer_names:
        metrics = train_optimizer(
            opt_name, train_loader, val_loader,
            dataset=dataset, n_epochs=n_epochs, device=device,
        )
        all_metrics.append(metrics)

    # dump everything to disk so we can run analysis.py independently
    output_path = os.path.join(results_dir, "raw_metrics.json")
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved → {output_path}")

    return all_metrics


if __name__ == "__main__":
    run_all_optimizers(dataset="cifar10", n_epochs=30)
