#!/usr/bin/env python3
"""
HEAT inference visualization per design.md.

Workflow:
1. Load best model and test set from splits.json
2. Run autoregressive rollout inference
3. Generate visualizations:
   - Training metrics plots (from metrics.csv)
   - Rollout GIFs (predicted, actual, diff)
   - Static comparison grids (subset of frames)

No subprocess calls - pure Python implementation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import patches as mpatches
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image
from tqdm import tqdm

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MORPH_ROOT = REPO_ROOT / "code" / "MORPH"
if str(MORPH_ROOT) not in sys.path:
    sys.path.insert(0, str(MORPH_ROOT))

# MORPH imports (lazy import in functions to avoid load overhead)


# ==================== LOGGING ====================

def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# ==================== CONSTANTS ====================

# Channel names from preprocess.py packing
CHANNEL_NAMES = [
    "Rcoord", "Zcoord",           # 0, 1 - spatial
    "vofm_maincharge",            # 2 - geometry
    "vofm_comp2",                 # 3 - booster (CYL) / striker (PLI)
    "vofm_wall",                  # 4 - wall
    "vofm_void",                  # 5 - void
    "av_density",                 # 6 - forces
    "av_pressure",                # 7 - forces (log1p applied)
    "speed",                      # 8 - forces (hypot)
]

# Which channels are geometry vs forces for visualization
GEOMETRY_CHANNELS = [2, 3, 4, 5]  # VoF masks
FORCE_CHANNELS = [6, 7, 8]        # density, pressure, speed

# Colormaps
CMAP_FIELD = "plasma"
CMAP_DIVERGING = "seismic"
CMAP_DIFF = "RdBu_r"  # Red-Blue diverging for differences


# ==================== METRICS PLOTTING ====================

def plot_training_metrics(metrics_csv: Path, output_dir: Path, logger: logging.Logger) -> List[Path]:
    """
    Plot training metrics from metrics.csv.
    
    Returns:
        List of output PNG paths
    """
    logger.info(f"Loading metrics from {metrics_csv}")
    
    # Parse CSV
    epochs = []
    train_loss = []
    val_loss = []
    val_mse = []
    val_mae = []
    val_rmse = []
    lr = []
    
    with open(metrics_csv) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                epochs.append(int(parts[0]))
                train_loss.append(float(parts[1]))
                val_loss.append(float(parts[2]))
                val_mse.append(float(parts[3]))
                val_mae.append(float(parts[4]))
                val_rmse.append(float(parts[5]))
                if len(parts) > 6:
                    lr.append(float(parts[6]))
    
    if not epochs:
        logger.warning("No metrics found in CSV")
        return []
    
    output_paths = []
    
    # Plot 1: Train vs Validation Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    
    out_path = output_dir / "metrics_loss.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(out_path)
    logger.info(f"Saved: {out_path}")
    
    # Plot 2: Validation Metrics Breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, val_mse, "r-", label="MSE", linewidth=2)
    ax.plot(epochs, val_mae, "g-", label="MAE", linewidth=2)
    ax.plot(epochs, val_rmse, "b-", label="RMSE", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.set_title("Validation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    out_path = output_dir / "metrics_validation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(out_path)
    logger.info(f"Saved: {out_path}")
    
    # Plot 3: Learning Rate Schedule
    if lr:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, lr, "g-", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        
        out_path = output_dir / "metrics_lr.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)
        logger.info(f"Saved: {out_path}")
    
    return output_paths


# ==================== MODEL LOADING ====================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
    model_size: str = "M",
    logger: logging.Logger = None,
) -> nn.Module:
    """Load MORPH model from checkpoint."""
    from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression  # type: ignore
    
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Model specs
    specs = {
        "Ti": {"dim": 256, "heads": 4, "depth": 4, "mlp_dim": 1024, "filters": 8},
        "S": {"dim": 512, "heads": 8, "depth": 4, "mlp_dim": 2048, "filters": 8},
        "M": {"dim": 768, "heads": 12, "depth": 8, "mlp_dim": 3072, "filters": 8},
        "L": {"dim": 1024, "heads": 16, "depth": 16, "mlp_dim": 4096, "filters": 8},
    }[model_size]
    
    model = ViT3DRegression(
        patch_size=8,
        dim=specs["dim"],
        depth=specs["depth"],
        heads=specs["heads"],
        heads_xa=32,
        mlp_dim=specs["mlp_dim"],
        max_components=3,
        conv_filter=specs["filters"],
        max_ar=1,
        max_patches=4096,
        max_fields=3,
        dropout=0.1,
        emb_dropout=0.1,
        lora_r_attn=0,
        lora_r_mlp=0,
        lora_alpha=None,
        lora_p=0.0,
        model_size=model_size,
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    # Remove module. prefix if present
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model loaded: {n_params:.2f}M parameters")
    
    return model


# ==================== DATA LOADING ====================

def load_test_data(
    data_npz: Path,
    splits_json: Path,
    modality: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load test set data.
    
    Returns:
        test_volume, test_timesteps, r_grid, z_grid, test_ids
    """
    logger.info(f"Loading data from {data_npz}")
    logger.info(f"Loading splits from {splits_json}")
    
    # Load splits
    with open(splits_json) as f:
        splits = json.load(f)
    
    splits_modality = splits.get("modality")
    if splits_modality and str(splits_modality).lower() != modality.lower():
        logger.warning(
            f"Splits modality is '{splits_modality}' but requested modality is '{modality}'"
        )

    test_ids = splits.get("test", [])
    logger.info(f"Test set: {len(test_ids)} instances")
    
    # Load NPZ
    data = np.load(data_npz)
    volume = data["volume"]  # (N, T, 9, H, W)
    timesteps = data["timesteps"]  # (N, T)
    instance_ids = data["instance_ids"]  # (N,)
    r_grid = data["r_grid"]  # (H, W)
    z_grid = data["z_grid"]  # (H, W)
    data.close()
    
    # Select test instances
    test_mask = np.isin(instance_ids, test_ids)
    test_volume = volume[test_mask]
    test_timesteps = timesteps[test_mask]
    test_ids_selected = instance_ids[test_mask].tolist()

    missing_test_ids = sorted(set(test_ids) - set(test_ids_selected))
    if missing_test_ids:
        logger.warning(
            f"{len(missing_test_ids)} test IDs from splits were not found in data: "
            f"{missing_test_ids[:5]}{'...' if len(missing_test_ids) > 5 else ''}"
        )
    
    logger.info(f"Test volume shape: {test_volume.shape}")
    
    return test_volume, test_timesteps, r_grid, z_grid, test_ids_selected


def normalize_volume(volume: np.ndarray, revin_dir: Path, modality: str) -> np.ndarray:
    """Normalize volume using RevIN."""
    from src.utils.normalization import RevIN  # type: ignore
    
    # Reshape to MORPH format (N, T, F, C, D, H, W)
    N, T, C, H, W = volume.shape
    volume_uptf7 = volume.reshape(N, T, 3, 3, 1, H, W)
    
    # Compute stats and normalize
    revin = RevIN(revin_dir)
    prefix = f"norm_{modality}_test"
    revin.compute_stats(volume_uptf7, prefix=prefix)
    volume_norm = revin.normalize(volume_uptf7, prefix=prefix)
    
    return volume_norm


# ==================== ROLLOUT INFERENCE ====================

def run_rollout(
    model: nn.Module,
    initial_frame: torch.Tensor,
    n_steps: int,
    device: torch.device,
    logger: logging.Logger,
) -> torch.Tensor:
    """
    Run autoregressive rollout.
    
    Args:
        model: MORPH model
        initial_frame: (F, C, D, H, W) first frame
        n_steps: number of steps to predict
        device: torch device
    
    Returns:
        predictions: (n_steps, F, C, D, H, W)
    """
    from src.utils.data_preparation_fast import FastARDataPreparer  # type: ignore
    
    logger.info(f"Running {n_steps}-step rollout")
    
    predictions = []
    current = initial_frame.unsqueeze(0).unsqueeze(0)  # (1, 1, F, C, D, H, W)
    
    with torch.no_grad():
        for step in tqdm(range(n_steps), desc="Rollout", leave=False):
            # Prepare AR input
            _, _, pred = model(current)  # pred: (1, F, C, D, H, W)
            predictions.append(pred.squeeze(0).cpu())
            
            # Next input is prediction
            current = pred.unsqueeze(1)  # (1, 1, F, C, D, H, W)
    
    return torch.stack(predictions, dim=0)  # (n_steps, F, C, D, H, W)


# ==================== VISUALIZATION ====================

def create_comparison_grid(
    actual: np.ndarray,
    predicted: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    channel_idx: int,
    frame_indices: List[int],
    output_path: Path,
    title: str = "Rollout Comparison",
    start_t: int = 1,
    logger: logging.Logger = None,
):
    """
    Create static grid comparing predicted vs actual for subset of timesteps.
    
    Args:
        actual: (T, H, W) actual data
        predicted: (T, H, W) predicted data
        r_grid, z_grid: coordinate grids
        channel_idx: which channel to visualize
        frame_indices: list of frame indices into actual/predicted arrays
        start_t: displayed timestep offset (default=1 since AR predicts t+1)
        output_path: output PNG path
    """
    logger = logger or logging.getLogger(__name__)
    
    n_frames = len(frame_indices)
    if n_frames == 0:
        logger.warning("No frame indices provided for comparison grid")
        return

    # Fixed 2xN layout: one horizontal strip of N frames (Actual on top, Predicted below).
    fig, axes = plt.subplots(2, n_frames, figsize=(3 * n_frames, 6), squeeze=False)
    
    # Get color limits from actual data
    vmin, vmax = np.percentile(actual, [2, 98])
    
    for col, t in enumerate(frame_indices):
        t_display = start_t + t

        # Actual
        ax_actual = axes[0, col]
        im = ax_actual.pcolormesh(r_grid, z_grid, actual[t], shading="auto", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
        ax_actual.set_title(f"T={t_display} (Actual)")
        ax_actual.set_aspect("equal")
        
        # Predicted
        ax_pred = axes[1, col]
        ax_pred.pcolormesh(r_grid, z_grid, predicted[t], shading="auto", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"T={t_display} (Predicted)")
        ax_pred.set_aspect("equal")
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=CHANNEL_NAMES[channel_idx])
    
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved comparison grid: {output_path}")


def create_rollout_gif(
    frames: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    output_path: Path,
    fps: float = 5.0,
    title: str = "Rollout",
    start_t: int = 1,
    logger: logging.Logger = None,
):
    """
    Create GIF from rollout frames.
    
    Args:
        frames: (T, H, W) array
        r_grid, z_grid: coordinate grids
        output_path: output GIF path
        fps: frames per second
        start_t: displayed timestep offset (default=1 since AR predicts t+1)
    """
    logger = logger or logging.getLogger(__name__)
    
    vmin, vmax = np.percentile(frames, [2, 98])
    duration_ms = int(1000 / fps)
    
    pil_frames = []
    for t in range(frames.shape[0]):
        t_display = start_t + t
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(r_grid, z_grid, frames[t], shading="auto", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} - T={t_display}")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax)
        
        # Convert to PIL
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        pil_frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close(fig)
    
    # Save GIF
    if pil_frames:
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        logger.info(f"Saved GIF: {output_path}")


def create_diff_gif(
    actual: np.ndarray,
    predicted: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    output_path: Path,
    fps: float = 5.0,
    title: str = "Difference",
    start_t: int = 1,
    logger: logging.Logger = None,
):
    """
    Create GIF showing difference between actual and predicted.
    
    Args:
        actual: (T, H, W)
        predicted: (T, H, W)
        r_grid, z_grid: coordinate grids
        output_path: output GIF path
        fps: frames per second
        start_t: displayed timestep offset (default=1 since AR predicts t+1)
    """
    logger = logger or logging.getLogger(__name__)
    
    diff = predicted - actual
    vmax = np.percentile(np.abs(diff), 98)
    vmin = -vmax
    duration_ms = int(1000 / fps)
    
    pil_frames = []
    for t in range(diff.shape[0]):
        t_display = start_t + t
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(r_grid, z_grid, diff[t], shading="auto", cmap=CMAP_DIFF, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} - T={t_display} (Pred - Actual)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="Difference")
        
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        pil_frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close(fig)
    
    if pil_frames:
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        logger.info(f"Saved diff GIF: {output_path}")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description="HEAT inference visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize metrics and run rollouts for test set
    python vis.py --modality cyl --checkpoint out/train/cyl/models/cyl_best.pth

    # Limit rollout length and frames
    python vis.py --modality cyl --checkpoint out/train/cyl/models/cyl_best.pth --max-rollout 20 --fps 10

    # Only plot metrics
    python vis.py --modality cyl --metrics-only
""",
    )
    
    parser.add_argument("--modality", type=str, required=True, choices=["cyl", "pli"],
                        help="Simulation modality")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to best model checkpoint")
    parser.add_argument("--data", type=Path, default=None,
                        help="Path to processed NPZ (default: processed/{modality}_data.npz)")
    parser.add_argument("--splits", type=Path, default=None,
                        help="Path to splits.json (default: out/splits.json)")
    parser.add_argument("--metrics", type=Path, default=None,
                        help="Path to metrics.csv (default: out/train/{modality}/metrics.csv)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out" / "infer",
                        help="Output directory for visualizations")
    parser.add_argument("--model-size", type=str, default="M", choices=["Ti", "S", "M", "L"],
                        help="Model size")
    parser.add_argument("--max-rollout", type=int, default=50,
                        help="Maximum rollout steps to visualize")
    parser.add_argument("--fps", type=float, default=5.0, help="GIF frames per second")
    parser.add_argument("--max-test-instances", type=int, default=5,
                        help="Max test instances to visualize")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Only plot training metrics, skip rollouts")
    parser.add_argument("--device-idx", type=int, default=0, help="CUDA device index")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    logger.info("=" * 60)
    logger.info("HEAT Inference Visualization")
    logger.info("=" * 60)
    
    # Resolve default paths
    if args.data is None:
        args.data = REPO_ROOT / "processed" / f"{args.modality}_data.npz"
    if args.splits is None:
        preferred_splits = REPO_ROOT / "out" / "train" / args.modality / "splits.json"
        legacy_splits = REPO_ROOT / "out" / "splits.json"
        if preferred_splits.exists():
            args.splits = preferred_splits
        else:
            args.splits = legacy_splits
    if args.metrics is None:
        args.metrics = REPO_ROOT / "out" / "train" / args.modality / "metrics.csv"
    
    # Output directories
    viz_dir = args.out_dir / "plots" / args.modality
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== METRICS PLOTTING ====================
    
    if args.metrics.exists():
        logger.info("Plotting training metrics")
        plot_training_metrics(args.metrics, viz_dir, logger)
    else:
        logger.warning(f"Metrics file not found: {args.metrics}")
    
    if args.metrics_only:
        logger.info("Metrics-only mode, skipping rollouts")
        return
    
    # ==================== ROLLOUT INFERENCE ====================
    
    # Check paths
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data not found: {args.data}")
    if not args.splits.exists():
        raise FileNotFoundError(f"Splits not found: {args.splits}")
    
    # Load test data
    test_volume, test_timesteps, r_grid, z_grid, test_ids = load_test_data(
        args.data, args.splits, args.modality, logger
    )
    
    # Limit test instances
    n_test = min(args.max_test_instances, len(test_ids))
    test_volume = test_volume[:n_test]
    test_timesteps = test_timesteps[:n_test]
    test_ids = test_ids[:n_test]
    logger.info(f"Visualizing {n_test} test instances")
    
    # Normalize
    logger.info("Normalizing test data")
    test_norm = normalize_volume(test_volume, args.out_dir / "revin_stats", args.modality)
    
    # Setup device and model
    from src.utils.device_manager import DeviceManager  # type: ignore
    devices = DeviceManager.list_devices()
    device = devices[args.device_idx] if devices else torch.device("cpu")
    logger.info(f"Device: {device}")
    
    model = load_model(args.checkpoint, device, args.model_size, logger)
    
    # ==================== ROLLOUT VISUALIZATION ====================
    
    for i, instance_id in enumerate(test_ids):
        logger.info(f"Processing instance {instance_id} ({i+1}/{n_test})")
        
        instance_dir = viz_dir / f"rollout_{instance_id}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        # Get volume for this instance
        vol = test_norm[i]  # (T, F, C, D, H, W)
        n_timesteps = vol.shape[0]
        
        # Limit rollout steps
        n_steps = min(args.max_rollout, n_timesteps - 1)
        
        # Initial frame
        initial = torch.from_numpy(vol[0]).float().to(device)  # (F, C, D, H, W)
        
        # Run rollout
        logger.info(f"  Running {n_steps}-step rollout")
        predictions = run_rollout(model, initial, n_steps, device, logger)
        
        # Get actual frames (starting from step 1)
        actual = torch.from_numpy(vol[1:n_steps+1]).float()  # (n_steps, F, C, D, H, W)
        
        # Reshape to (T, 9, H, W) for visualization
        pred_np = predictions.numpy().reshape(n_steps, 9, r_grid.shape[0], r_grid.shape[1])
        actual_np = actual.numpy().reshape(n_steps, 9, r_grid.shape[0], r_grid.shape[1])
        
        # Visualize each force channel
        for ch in FORCE_CHANNELS:
            ch_name = CHANNEL_NAMES[ch]
            logger.info(f"  Visualizing channel: {ch_name}")
            
            # Extract channel
            pred_ch = pred_np[:, ch, :, :]
            actual_ch = actual_np[:, ch, :, :]
            
            # Comparison grid: 5-6 frames laid out horizontally in a single row.
            n_grid_frames = min(6, n_steps)
            if n_grid_frames > 1:
                grid_frames = np.linspace(0, n_steps - 1, n_grid_frames, dtype=int).tolist()
            else:
                grid_frames = [0]
            create_comparison_grid(
                actual_ch, pred_ch, r_grid, z_grid, ch,
                grid_frames, instance_dir / f"{ch_name}_comparison.png",
                f"{instance_id} - {ch_name}", start_t=1, logger=logger
            )
            
            # GIF of prediction
            create_rollout_gif(
                pred_ch, r_grid, z_grid,
                instance_dir / f"{ch_name}_predicted.gif",
                args.fps, f"{instance_id} - {ch_name} (Predicted)", start_t=1, logger=logger
            )
            
            # GIF of actual
            create_rollout_gif(
                actual_ch, r_grid, z_grid,
                instance_dir / f"{ch_name}_actual.gif",
                args.fps, f"{instance_id} - {ch_name} (Actual)", start_t=1, logger=logger
            )
            
            # GIF of diff
            create_diff_gif(
                actual_ch, pred_ch, r_grid, z_grid,
                instance_dir / f"{ch_name}_diff.gif",
                args.fps, f"{instance_id} - {ch_name}", start_t=1, logger=logger
            )
    
    logger.info("=" * 60)
    logger.info("Visualization complete!")
    logger.info(f"Output: {viz_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
