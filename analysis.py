"""
analysis.py — Post-training analysis: convergence plots, efficiency frontier,
and summary statistics for all six optimizers.

Reads results/raw_metrics.json produced by train.py. Can be re-run as many
times as needed without re-training.

Outputs : results/convergence_loss.png
          results/convergence_acc.png
          results/gradient_norm.png
          results/efficiency_frontier.png
          results/convergence_speed.png
          results/optimizer_comparison.csv
Usage   : python analysis.py
"""

import os
import json
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Plot style ────────────────────────────────────────────────────────────────

# not using seaborn to keep dependencies minimal, but we still want clean plots
COLORS = {
    "SGD"     : "#e05c4b",
    "SGD+Mom" : "#e0944b",
    "Adam"    : "#4b7be0",
    "RMSprop" : "#4bc0e0",
    "Adagrad" : "#7b4be0",
    "LBFGS"   : "#4be07b",
}

LINESTYLES = {
    "SGD"     : "-",
    "SGD+Mom" : "--",
    "Adam"    : "-",
    "RMSprop" : "--",
    "Adagrad" : "-.",
    "LBFGS"   : ":",
}

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 10,
    "axes.titlesize"  : 12,
    "axes.labelsize"  : 11,
    "legend.fontsize" : 9,
    "figure.dpi"      : 120,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
})

RESULTS_DIR = "results"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_metrics(path: str = None) -> list[dict]:
    if path is None:
        path = os.path.join(RESULTS_DIR, "raw_metrics.json")
    with open(path) as f:
        return json.load(f)


# ── Plot 1: Training loss convergence ─────────────────────────────────────────

def plot_loss_convergence(all_metrics: list[dict], out_dir: str = RESULTS_DIR):
    fig, ax = plt.subplots(figsize=(9, 5))

    for m in all_metrics:
        name = m["optimizer"]
        epochs = range(1, len(m["train_loss"]) + 1)
        ax.plot(
            epochs, m["train_loss"],
            color=COLORS[name], linestyle=LINESTYLES[name],
            linewidth=1.8, label=name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Convergence Curves — Training Loss")
    ax.legend(loc="upper right", framealpha=0.7)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    path = os.path.join(out_dir, "convergence_loss.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 2: Validation accuracy ───────────────────────────────────────────────

def plot_val_accuracy(all_metrics: list[dict], out_dir: str = RESULTS_DIR):
    fig, ax = plt.subplots(figsize=(9, 5))

    for m in all_metrics:
        name = m["optimizer"]
        epochs = range(1, len(m["val_acc"]) + 1)
        ax.plot(
            epochs, [v * 100 for v in m["val_acc"]],
            color=COLORS[name], linestyle=LINESTYLES[name],
            linewidth=1.8, label=name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy vs. Epoch")
    ax.legend(loc="lower right", framealpha=0.7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    path = os.path.join(out_dir, "convergence_acc.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 3: Gradient norm ─────────────────────────────────────────────────────

def plot_gradient_norm(all_metrics: list[dict], out_dir: str = RESULTS_DIR):
    """
    LBFGS doesn't expose per-step grad norms the same way, so we skip it here.
    The plot is most interesting for SGD vs momentum vs Adam.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for m in all_metrics:
        name = m["optimizer"]
        if name == "LBFGS":
            continue  # closure-based, norms are not tracked the same way
        epochs = range(1, len(m["grad_norm"]) + 1)
        ax.plot(
            epochs, m["grad_norm"],
            color=COLORS[name], linestyle=LINESTYLES[name],
            linewidth=1.8, label=name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Gradient Norm  $\|\nabla L\|_2$")
    ax.set_title("Gradient Norm Over Training")
    ax.legend(loc="upper right", framealpha=0.7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    path = os.path.join(out_dir, "gradient_norm.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 4: Efficiency frontier ───────────────────────────────────────────────

def plot_efficiency_frontier(all_metrics: list[dict], out_dir: str = RESULTS_DIR):
    """
    Final val acc vs total training time — the 'bang for your compute buck'
    view. Points in the top-left are better (fast and accurate).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for m in all_metrics:
        name      = m["optimizer"]
        total_sec = sum(m["epoch_time_sec"])
        final_acc = m["val_acc"][-1] * 100

        ax.scatter(total_sec, final_acc, color=COLORS[name], s=120, zorder=5)
        # nudge the label slightly so they don't overlap
        ax.annotate(
            name,
            xy=(total_sec, final_acc),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9, color=COLORS[name],
        )

    ax.set_xlabel("Total Training Time (seconds)")
    ax.set_ylabel("Final Validation Accuracy (%)")
    ax.set_title("Efficiency Frontier — Accuracy vs. Compute Cost")
    ax.grid(alpha=0.25, linewidth=0.6)

    path = os.path.join(out_dir, "efficiency_frontier.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 5: Epochs to 90% of best accuracy ────────────────────────────────────

def plot_gradient_sparsity(all_metrics: list[dict], out_dir: str = RESULTS_DIR):
    """
    Gradient sparsity over training — what fraction of gradient elements are
    near-zero (<1e-4 in absolute value)? Higher sparsity means the gradient
    is more compressible, which matters for distributed training with
    methods like PowerSGD.

    This plot is the most directly research-relevant one in the set.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for m in all_metrics:
        name = m["optimizer"]
        if "grad_sparsity" not in m or not m["grad_sparsity"]:
            continue
        epochs = range(1, len(m["grad_sparsity"]) + 1)
        ax.plot(
            epochs, [s * 100 for s in m["grad_sparsity"]],
            color=COLORS[name], linestyle=LINESTYLES[name],
            linewidth=1.8, label=name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Sparsity (%)")
    ax.set_title("Gradient Sparsity Over Training\n"
                 r"(fraction of $|\nabla L_i| < 10^{-4}$  — higher = more compressible)")
    ax.legend(loc="lower right", framealpha=0.7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    path = os.path.join(out_dir, "gradient_sparsity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_convergence_speed(all_metrics: list[dict], out_dir: str = RESULTS_DIR):
    """
    How many epochs until each optimizer reaches 90% of the best final
    accuracy across all runs? Lower = faster convergence.
    """
    best_overall = max(m["val_acc"][-1] for m in all_metrics)
    target = 0.9 * best_overall

    names  = []
    epochs_to_target = []

    for m in all_metrics:
        name = m["optimizer"]
        reached = next(
            (i + 1 for i, acc in enumerate(m["val_acc"]) if acc >= target),
            len(m["val_acc"])  # never reached — use max epochs
        )
        names.append(name)
        epochs_to_target.append(reached)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        names, epochs_to_target,
        color=[COLORS[n] for n in names],
        edgecolor="white", linewidth=0.5,
    )

    for bar, val in zip(bars, epochs_to_target):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Epochs to 90% of Best Accuracy")
    ax.set_title("Convergence Speed Comparison")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    path = os.path.join(out_dir, "convergence_speed.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    return dict(zip(names, epochs_to_target))


# ── CSV summary table ─────────────────────────────────────────────────────────

def save_summary_csv(all_metrics: list[dict], epochs_to_90: dict, out_dir: str = RESULTS_DIR):
    path = os.path.join(out_dir, "optimizer_comparison.csv")
    fieldnames = [
        "optimizer", "final_loss", "final_val_acc",
        "epochs_to_90pct", "total_time_sec", "gradient_evals",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            name = m["optimizer"]
            writer.writerow({
                "optimizer"      : name,
                "final_loss"     : round(m["train_loss"][-1], 5),
                "final_val_acc"  : round(m["val_acc"][-1], 5),
                "epochs_to_90pct": epochs_to_90.get(name, "N/A"),
                "total_time_sec" : round(sum(m["epoch_time_sec"]), 1),
                "gradient_evals" : m["grad_evals"][-1],
            })

    print(f"Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_analysis(metrics_path: str = None, out_dir: str = RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    all_metrics = load_metrics(metrics_path)

    print(f"Loaded metrics for: {[m['optimizer'] for m in all_metrics]}")
    print()

    plot_loss_convergence(all_metrics, out_dir)
    plot_val_accuracy(all_metrics, out_dir)
    plot_gradient_norm(all_metrics, out_dir)
    plot_gradient_sparsity(all_metrics, out_dir)
    plot_efficiency_frontier(all_metrics, out_dir)
    epochs_to_90 = plot_convergence_speed(all_metrics, out_dir)
    save_summary_csv(all_metrics, epochs_to_90, out_dir)

    print("\nAll plots generated.")


if __name__ == "__main__":
    run_analysis()
