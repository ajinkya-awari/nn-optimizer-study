"""
run_all.py — Convenience wrapper for running the full pipeline.
Identical to: python main.py --dataset cifar10 --epochs 30

Usage: python run_all.py
"""

from main import main
main(dataset="cifar10", epochs=30)
