"""
main.py — Full experiment pipeline. Trains all six optimizers, runs analysis,
and generates the loss landscape for the Adam checkpoint.

This is the top-level entry point — running this file does everything in order.
Estimated runtime: 1–3 hours on CPU, 20–40 min with a GPU.

Outputs : everything in results/ and checkpoints/
Usage   : python main.py
          python main.py --dataset mnist --epochs 20   (faster)
"""

import argparse
import time

from train    import run_all_optimizers
from analysis import run_analysis
from loss_landscape import run_landscape


def main(dataset: str, epochs: int):
    t0 = time.time()

    print("=" * 60)
    print(" Neural Network Optimizer Comparison Study")
    print("=" * 60)
    print(f" Dataset : {dataset.upper()}")
    print(f" Epochs  : {epochs}")
    print("=" * 60)
    print()

    # ── Phase 1: train ────────────────────────────────────────────────────────
    print("Phase 1/3 — Training\n")
    run_all_optimizers(dataset=dataset, n_epochs=epochs)

    # ── Phase 2: analysis ─────────────────────────────────────────────────────
    print("\nPhase 2/3 — Analysis\n")
    run_analysis()

    # ── Phase 3: loss landscape ───────────────────────────────────────────────
    # Adam tends to find a flatter, more interesting basin — good for the visualisation
    print("\nPhase 3/3 — Loss Landscape (Adam checkpoint)\n")
    try:
        run_landscape(optimizer_name="Adam", dataset=dataset)
    except FileNotFoundError as e:
        print(f"Skipping landscape: {e}")

    elapsed = time.time() - t0
    print(f"\nAll done. Total time: {elapsed/60:.1f} min")
    print("Results are in the results/ folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimizer comparison study")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"],
                        help="Dataset to train on (default: cifar10)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    args = parser.parse_args()

    main(args.dataset, args.epochs)
