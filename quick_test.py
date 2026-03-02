"""
quick_test.py — Smoke test that finishes in ~5 minutes.

Uses MNIST with 5 epochs instead of the full CIFAR-10 / 30-epoch run.
Good for checking the environment is set up correctly before committing
to the full experiment.

Outputs : results_quicktest/ (separate folder so it doesn't overwrite real results)
Usage   : python quick_test.py
"""

import os
import json

from train    import run_all_optimizers
from analysis import run_analysis

OUT_DIR = "results_quicktest"

print("Running quick test — MNIST, 5 epochs per optimizer")
print("This should finish in ~5 minutes on CPU.\n")

os.makedirs(OUT_DIR, exist_ok=True)

all_metrics = run_all_optimizers(
    dataset     = "mnist",
    n_epochs    = 5,
    batch_size  = 256,
    results_dir = OUT_DIR,
)

run_analysis(
    metrics_path = os.path.join(OUT_DIR, "raw_metrics.json"),
    out_dir      = OUT_DIR,
)

print(f"\nQuick test complete. Check {OUT_DIR}/ for outputs.")
print("If everything looks right, run: python main.py")
