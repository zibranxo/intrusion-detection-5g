"""
train_autoencoder.py — Train the trajectory autoencoder for anomaly detection.

Generates synthetic "normal" trajectory data, trains a lightweight
autoencoder (<2K params), computes the anomaly threshold, and exports
to ONNX for edge inference in the pipeline.

This is a self-contained script — no pre-collected data is needed.
After training, pass --learned-ids to quantized_pipeline.py to use
the learned anomaly detector alongside (or replacing) geometric rules.

Usage:
    python train_autoencoder.py
    python train_autoencoder.py --samples 20000 --epochs 100 --output models/ae.onnx
"""

import argparse
from pathlib import Path

import numpy as np

from config import INPUT_SIZE, RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train trajectory autoencoder for anomaly detection."
    )
    parser.add_argument(
        "--samples", type=int, default=10000,
        help="Number of synthetic normal samples to generate",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.01,
        help="Gaussian noise std for synthetic data",
    )
    parser.add_argument(
        "--threshold-percentile", type=float, default=95.0,
        help="Percentile of normal recon error to use as anomaly threshold",
    )
    parser.add_argument(
        "--output", default="models/trajectory_ae.onnx",
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Training device",
    )
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from utils.autoencoder import (
        _build_torch_autoencoder,
        compute_anomaly_threshold,
        export_to_onnx,
        generate_synthetic_trajectories,
        train_autoencoder,
    )

    print("═" * 60)
    print("  Trajectory Autoencoder Training")
    print("═" * 60)

    # ── Generate synthetic training data ──────────────────────────────────
    print(f"\n[1/4] Generating {args.samples} synthetic normal samples...")
    features = generate_synthetic_trajectories(
        n_samples=args.samples,
        frame_shape=(INPUT_SIZE[1], INPUT_SIZE[0]),
        noise_std=args.noise_std,
    )
    print(f"       Feature shape: {features.shape}  "
          f"range: [{features.min():.2f}, {features.max():.2f}]")

    # Train / validation split
    split = int(len(features) * 0.85)
    train_data = features[:split]
    val_data = features[split:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data)),
        batch_size=128, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_data)),
        batch_size=128, shuffle=False,
    )

    # ── Build and train ──────────────────────────────────────────────────
    print(f"\n[2/4] Training autoencoder ({args.epochs} epochs, {args.lr} lr)...")
    model = _build_torch_autoencoder()

    # Print parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       Parameters: {n_params}")

    losses = train_autoencoder(
        model, train_loader,
        epochs=args.epochs, lr=args.lr, device=args.device,
    )
    print(f"       Final loss: {losses[-1]:.6f}")

    # ── Compute anomaly threshold ─────────────────────────────────────────
    print(f"\n[3/4] Computing anomaly threshold (p{args.threshold_percentile:.0f})...")
    threshold = compute_anomaly_threshold(
        model, val_loader,
        percentile=args.threshold_percentile,
        device=args.device,
    )

    # Save threshold alongside the model
    threshold_path = Path(args.output).with_suffix(".threshold.txt")
    threshold_path.write_text(f"{threshold:.8f}\n")
    print(f"       Threshold saved to {threshold_path}")

    # ── Export to ONNX ────────────────────────────────────────────────────
    print(f"\n[4/4] Exporting to ONNX → {args.output}")
    export_to_onnx(model, args.output, device=args.device)

    # ── Summary ───────────────────────────────────────────────────────────
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\n{'═' * 60}")
    print(f"  Training complete!")
    print(f"  Model:       {args.output}  ({size_mb:.2f} MB)")
    print(f"  Threshold:   {threshold:.6f}")
    print(f"  Parameters:  {n_params}")
    print(f"  Final loss:  {losses[-1]:.6f}")
    print(f"\n  Next: python quantized_pipeline.py --learned-ids --source test.mp4")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
