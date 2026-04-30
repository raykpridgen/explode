#!/usr/bin/env python3
"""
Train MORPH surrogate on HEAT data per design.md contracts.

Standalone script - no subprocess invocations. Imports from MORPH via sys.path.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ==================== PATH SETUP ====================

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# MORPH root (default: code/MORPH)
MORPH_ROOT = REPO_ROOT / "code" / "MORPH"
if str(MORPH_ROOT) not in sys.path:
    sys.path.insert(0, str(MORPH_ROOT))

# ==================== LOGGING ====================

def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


# ==================== CONFIGURATION ====================

# Model specs per design.md: Medium (M) for both CYL and PLI
MORPH_MODELS = {
    "Ti": {"patch_size": 8, "dim": 256, "heads": 4, "depth": 4, "mlp_dim": 1024, "filters": 8},
    "S": {"patch_size": 8, "dim": 512, "heads": 8, "depth": 4, "mlp_dim": 2048, "filters": 8},
    "M": {"patch_size": 8, "dim": 768, "heads": 12, "depth": 8, "mlp_dim": 3072, "filters": 8},
    "L": {"patch_size": 8, "dim": 1024, "heads": 16, "depth": 16, "mlp_dim": 4096, "filters": 8},
}

# FM checkpoint filenames per plan.md
FM_CHECKPOINTS = {
    "Ti": "morph-Ti-FM-max_ar1_ep225.pth",
    "S": "morph-S-FM-max_ar1_ep225.pth",
    "M": "morph-M-FM-max_ar1_ep290_latestbatch.pth",
    "L": "morph-L-FM-max_ar16_ep189_latestbatch.pth",
}


# ==================== DATASET ====================

class ARWindowDataset(Dataset):
    """
    Memory-efficient dataset that generates AR windows on-the-fly with per-instance normalization.
    
    Instead of materializing all windows (which explodes memory),
    this computes windows dynamically and applies per-instance normalization on-the-fly.
    """
    
    def __init__(
        self,
        volume: np.ndarray,  # (N, T, F, C, D, H, W) - unnormalized
        ar_order: int = 1,
        dtype: torch.dtype = torch.float32,
        normalize: bool = True,
        eps: float = 1e-5,
    ):
        """
        Args:
            volume: Raw volume array (N, T, F, C, D, H, W) - NOT pre-normalized
            ar_order: Autoregressive order (default: 1)
            dtype: Torch dtype for output tensors
            normalize: Whether to apply per-instance normalization
            eps: Epsilon for numerical stability in normalization
        """
        self.volume = volume
        self.ar_order = ar_order
        self.dtype = dtype
        self.normalize = normalize
        self.eps = eps
        
        # Compute windows per instance
        self.n_instances = volume.shape[0]
        self.n_timesteps = volume.shape[1]
        self.n_fields = volume.shape[2]
        self.n_components = volume.shape[3]
        self.n_windows_per_instance = self.n_timesteps - ar_order
        
        # Total windows
        self.total_windows = self.n_instances * self.n_windows_per_instance
        
        # Pre-compute per-instance statistics if normalizing
        if self.normalize:
            logger.info(f"Computing per-instance statistics for {self.n_instances} instances...")
            self.instance_stats = self._compute_instance_stats()
        else:
            self.instance_stats = None
    
    def _compute_instance_stats(self) -> List[Dict]:
        """Compute mean and std for each instance across time dimension."""
        stats = []
        for i in range(self.n_instances):
            instance_data = self.volume[i]  # (T, F, C, D, H, W)
            # Compute stats per (F, C) across time, D, H, W
            mean = np.mean(instance_data, axis=(0, 3, 4, 5), keepdims=True)  # (1, F, C, 1, 1, 1)
            std = np.std(instance_data, axis=(0, 3, 4, 5), keepdims=True) + self.eps
            stats.append({"mean": mean, "std": std})
        return stats
    
    def _normalize_instance(self, instance_idx: int, data: np.ndarray) -> np.ndarray:
        """Normalize data for a specific instance."""
        if not self.normalize or self.instance_stats is None:
            return data
        stats = self.instance_stats[instance_idx]
        return (data - stats["mean"]) / stats["std"]
        
    def __len__(self) -> int:
        return self.total_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate window on-the-fly with normalization.
        
        Args:
            idx: Global window index
            
        Returns:
            (x, y) where x is input window, y is target (both normalized)
        """
        # Map global index to (instance, timestep)
        instance_idx = idx // self.n_windows_per_instance
        local_window_idx = idx % self.n_windows_per_instance
        t_start = local_window_idx
        
        # Extract window
        # x: (ar_order, F, C, D, H, W)
        x = self.volume[instance_idx, t_start:t_start + self.ar_order].copy()
        
        # y: (F, C, D, H, W) - target is next frame
        y = self.volume[instance_idx, t_start + self.ar_order].copy()
        
        # Apply per-instance normalization
        if self.normalize:
            x = self._normalize_instance(instance_idx, x)
            y = self._normalize_instance(instance_idx, y)
        
        # Convert to torch tensors
        x_tensor = torch.from_numpy(x).to(self.dtype)
        y_tensor = torch.from_numpy(y).to(self.dtype)
        
        return x_tensor, y_tensor


# Legacy dataset for precomputed windows (kept for eval mode if needed)
class ARDataset(Dataset):
    """Dataset for precomputed autoregressive (X, y) pairs."""
    
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
    
    def __len__(self) -> int:
        return self.x.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


# ==================== UTILITIES ====================

def split_instances(n_instances: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split instance indices into train/val/test per design.md (80/10/10).
    
    Returns:
        train_indices, val_indices, test_indices
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_instances)
    rng.shuffle(indices)
    
    n_train = int(0.8 * n_instances)
    n_val = int(0.1 * n_instances)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return train_idx, val_idx, test_idx


def save_splits(
    output_path: Path,
    modality: str,
    seed: int,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> None:
    """Save splits.json per design.md."""
    splits = {
        "modality": modality,
        "split_seed": seed,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
    }
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)


class WarmupCosineScheduler:
    """Warmup + cosine decay LR scheduler per design.md."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self) -> float:
        """Update LR and return current value."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        return lr


# ==================== METRICS ====================

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute MSE, MAE, RMSE in normalized space."""
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    rmse = math.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM using MORPH Metrics3DCalculator."""
    from src.utils.metrics_3d import Metrics3DCalculator  # type: ignore
    return Metrics3DCalculator.calculate_ssim(pred, target).item()


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[WarmupCosineScheduler],
    step_checkpoint_path: Path,
    save_freq: int,
    global_step: int,
    best_val_loss: float,
    logger: logging.Logger,
) -> Tuple[float, int, float]:
    """
    Train for one epoch.
    
    Returns:
        avg_loss, updated_global_step, updated_best_val_loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (xb, yb) in enumerate(pbar):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        _, _, pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
        global_step += 1
        
        # Save step checkpoint periodically
        if save_freq > 0 and global_step % save_freq == 0:
            ckpt = {
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            torch.save(ckpt, step_checkpoint_path)
        
        pbar.set_postfix({"loss": f"{loss.item():.6f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
    
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, global_step, best_val_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[float, Dict[str, float]]:
    """
    Validate model.
    
    Returns:
        avg_loss, metrics_dict
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_metrics: Dict[str, List[float]] = {"mse": [], "mae": [], "rmse": []}
    
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Validation", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            _, _, pred = model(xb)
            loss = criterion(pred, yb)
            
            total_loss += loss.item()
            n_batches += 1
            
            batch_metrics = compute_metrics(pred, yb)
            for k, v in batch_metrics.items():
                all_metrics[k].append(v)
    
    avg_loss = total_loss / max(n_batches, 1)
    metrics = {k: sum(v) / max(len(v), 1) for k, v in all_metrics.items()}
    metrics["loss"] = avg_loss
    
    return avg_loss, metrics


def test_evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Evaluate on test set with full metrics."""
    model.eval()
    
    all_metrics: Dict[str, List[float]] = {
        "mse": [], "mae": [], "rmse": [], "ssim": []
    }
    
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Testing", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            _, _, pred = model(xb)
            
            batch_metrics = compute_metrics(pred, yb)
            for k, v in batch_metrics.items():
                all_metrics[k].append(v)
            
            # SSIM is expensive, compute on subset
            if xb.shape[0] <= 8:  # Only for small batches
                try:
                    ssim_val = compute_ssim(pred, yb)
                    all_metrics["ssim"].append(ssim_val)
                except Exception:
                    pass
    
    metrics = {k: sum(v) / max(len(v), 1) for k, v in all_metrics.items()}
    return metrics


# ==================== MAIN ====================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MORPH surrogate on HEAT data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train CYL from scratch with Medium model (memory-efficient streaming)
    python train_explode.py --modality cyl --data processed/cyl_data.npz --model-size M

    # Reduce memory further with smaller batch size (for large grids like PLI 1120x400)
    python train_explode.py --modality pli --data processed/pli_data.npz --batch-size 2

    # Resume training from checkpoint
    python train_explode.py --modality pli --data processed/pli_data.npz --resume out/train/pli/models/pli_checkpoint.pth

    # Evaluation only
    python train_explode.py --modality cyl --data processed/cyl_data.npz --eval-only --resume out/train/cyl/models/cyl_best.pth

Memory efficiency:
    This script uses streaming AR window generation with per-instance normalization
    to avoid materializing large intermediate arrays.
    
    For large grids (e.g., PLI 1120x400), use --batch-size 2 or --batch-size 4.
    
    Memory estimates:
    - CYL (560x200): ~8-15GB with batch-size 8
    - PLI (1120x400): ~15-25GB with batch-size 2-4

Output structure:
    out/
        splits.json              # Train/val/test instance IDs
        train/
            {modality}/
                metrics.csv      # Per-epoch training log
                models/
                    {modality}_best.pth     # Best model only
""",
    )
    
    # Data args
    parser.add_argument("--modality", type=str, required=True, choices=["cyl", "pli"],
                        help="Simulation modality")
    parser.add_argument("--data", type=Path, required=True, help="Path to processed NPZ")
    parser.add_argument("--morph-root", type=Path, default=MORPH_ROOT,
                        help="Path to MORPH repository")
    
    # Model args
    parser.add_argument("--model-size", type=str, default="M", choices=["Ti", "S", "M", "L"],
                        help="Model size (default: M per design.md)")
    parser.add_argument("--heads-xa", type=int, default=32, help="Cross-attention heads")
    parser.add_argument("--ar-order", type=int, default=1, help="Autoregressive order")
    parser.add_argument("--max-ar-order", type=int, default=1, help="Max AR order for model")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum LR for cosine decay")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup-frac", type=float, default=0.1,
                        help="Fraction of steps for warmup (default: 0.1)")
    
    # Checkpoint args
    parser.add_argument("--save-freq", type=int, default=100,
                        help="Save step checkpoint every N steps (0 to disable)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluation mode only (requires --resume)")
    
    # Data loading args
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for splits")
    
    # Output args
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out",
                        help="Output directory root")
    parser.add_argument("--device-idx", type=int, default=0, help="CUDA device index")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Validate eval-only
    if args.eval_only and not args.resume:
        parser.error("--eval-only requires --resume")
    
    # Setup paths
    modality = args.modality
    model_size = args.model_size
    train_dir = args.out_dir / "train" / modality
    model_dir = train_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = train_dir / "train.log" if not args.eval_only else None
    logger = setup_logging(args.verbose, log_file)
    
    logger.info("=" * 60)
    logger.info("HEAT × MORPH Training")
    logger.info(f"Modality: {modality}, Model: {model_size}")
    logger.info("=" * 60)
    
    # Ensure MORPH on path
    morph_root = args.morph_root.resolve()
    if str(morph_root) not in sys.path:
        sys.path.insert(0, str(morph_root))
        logger.debug(f"Added {morph_root} to sys.path")
    
    # Import MORPH modules
    try:
        from src.utils.data_preparation_fast import FastARDataPreparer  # type: ignore
        from src.utils.device_manager import DeviceManager  # type: ignore
        from src.utils.select_fine_tuning_parameters import SelectFineTuningParameters  # type: ignore
        from src.utils.trainers import Trainer  # type: ignore
        from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression  # type: ignore
        logger.debug("MORPH imports successful")
    except ImportError as e:
        logger.error(f"Failed to import MORPH modules: {e}")
        logger.error(f"sys.path: {sys.path}")
        raise
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    data_npz = np.load(args.data)
    volume = data_npz["volume"]  # (N, T, 9, H, W) - packed channels
    instance_ids = data_npz["instance_ids"]
    timesteps = data_npz["timesteps"]  # (N, T)
    valid_mask = data_npz.get("valid_mask", np.ones_like(timesteps, dtype=bool))
    
    n_instances, max_t, n_channels, H, W = volume.shape
    logger.info(f"Data shape: {volume.shape}")
    logger.info(f"Instances: {n_instances}, Max timesteps: {max_t}")
    logger.info(f"Grid: {H}x{W}, Channels: {n_channels}")
    
    # Split instances
    train_idx, val_idx, test_idx = split_instances(n_instances, args.split_seed)
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    # Save splits.json
    splits_path = args.out_dir / "splits.json"
    save_splits(
        splits_path,
        modality,
        args.split_seed,
        instance_ids[train_idx].tolist(),
        instance_ids[val_idx].tolist(),
        instance_ids[test_idx].tolist(),
    )
    logger.info(f"Saved splits to {splits_path}")
    
    # Prepare volume for MORPH: (N, T, 9, H, W) -> (N, T, F, C, D, H, W)
    # We have 9 channels = 3 fields × 3 components
    # Reshape: (N, T, 9, H, W) -> (N, T, 3, 3, 1, H, W) assuming D=1 (2D data)
    D = 1
    volume_uptf7 = volume.reshape(n_instances, max_t, 3, 3, D, H, W)
    volume_uptf7 = volume_uptf7.transpose(0, 1, 2, 3, 4, 5, 6)  # (N, T, F, C, D, H, W)
    volume_uptf7 = volume_uptf7.astype(np.float32)
    
    # Split data
    train_vol = volume_uptf7[train_idx]
    val_vol = volume_uptf7[val_idx]
    test_vol = volume_uptf7[test_idx]
    
    logger.info(f"Train volume: {train_vol.shape}")
    logger.info(f"Val volume: {val_vol.shape}")
    logger.info(f"Test volume: {test_vol.shape}")
    
    # Memory-efficient approach: Use per-instance normalization in streaming dataset
    # instead of MORPH RevIN which creates large intermediate arrays
    logger.info("Using per-instance normalization in streaming DataLoader")
    logger.info("(This avoids materializing large normalized arrays in memory)")
    
    # Clean up intermediate arrays to save memory
    logger.info("Cleaning up intermediate arrays...")
    del volume, volume_uptf7
    import gc
    gc.collect()
    
    # Prepare autoregressive windows using streaming dataset with on-the-fly normalization
    # This generates windows on-the-fly and normalizes per-instance to avoid memory explosion
    logger.info(f"Preparing streaming AR windows (ar_order={args.ar_order})")
    
    # Create streaming datasets with per-instance normalization
    # Normalization is computed per-instance inside the DataLoader workers
    train_dataset = ARWindowDataset(train_vol, ar_order=args.ar_order, normalize=True)
    
    if val_vol.shape[0] > 0:
        val_dataset = ARWindowDataset(val_vol, ar_order=args.ar_order, normalize=True)
    else:
        # Create dataset from train for validation if no val split
        val_dataset = None
        logger.warning("No validation split, using training metrics only")
    
    test_dataset = ARWindowDataset(test_vol, ar_order=args.ar_order, normalize=True)
    
    # Free the split volumes - they're now referenced inside the datasets
    # The dataset keeps them, but normalizes on-the-fly
    # We can optionally delete them if memory is still tight, but dataset needs them
    logger.info("Dataset created with per-instance normalization")
    logger.info(f"Memory per instance: {train_vol[0].nbytes / 1e9:.2f} GB")
    
    logger.info(f"AR windows: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}, test={len(test_dataset)}")
    
    # Setup device
    devices = DeviceManager.list_devices()
    device = devices[args.device_idx] if devices else torch.device("cpu")
    logger.info(f"Device: {device}")
    
    # Create model
    model_spec = MORPH_MODELS[model_size]
    model = ViT3DRegression(
        patch_size=model_spec["patch_size"],
        dim=model_spec["dim"],
        depth=model_spec["depth"],
        heads=model_spec["heads"],
        heads_xa=args.heads_xa,
        mlp_dim=model_spec["mlp_dim"],
        max_components=3,
        conv_filter=model_spec["filters"],
        max_ar=args.max_ar_order,
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
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {n_params:.2f}M")
    
    # Configure optimizer (full finetune level 4 per design.md)
    ft_args = SimpleNamespace(
        ft_level4=True,
        ft_level1=False,
        ft_level2=False,
        ft_level3=False,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_level4=args.lr,
        wd_level4=args.weight_decay,
        rank_lora_attn=0,
        rank_lora_mlp=0,
        lora_p=0.0,
    )
    selector = SelectFineTuningParameters(model, ft_args)
    optimizer = selector.configure_levels()
    logger.info(f"Optimizer: AdamW, LR={args.lr}, WD={args.weight_decay}")
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0
    
    if args.resume:
        logger.info(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        
        state_dict = ckpt.get("model_state_dict", ckpt)
        if state_dict and next(iter(state_dict)).startswith("module."):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=True)
        
        if "optimizer_state_dict" in ckpt and not args.eval_only:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("Loaded optimizer state")
        
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    
    # Create dataloaders using streaming datasets
    logger.info("Creating DataLoaders with streaming datasets...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset if val_dataset is not None else train_dataset,  # Fallback if no val
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Evaluation only mode
    if args.eval_only:
        logger.info("Running evaluation only")
        test_metrics = test_evaluate(model, test_loader, device, logger)
        logger.info(f"Test metrics: {test_metrics}")
        
        # Save test metrics
        test_metrics_path = args.out_dir / "infer" / "metrics" / f"{modality}_test_metrics.json"
        test_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Saved test metrics to {test_metrics_path}")
        return
    
    # Setup warmup + cosine scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, args.lr, args.min_lr)
    logger.info(f"LR schedule: warmup={warmup_steps} steps, total={total_steps} steps")
    
    # Checkpoint paths
    step_ckpt_path = model_dir / f"{modality}_step.pth"
    best_ckpt_path = model_dir / f"{modality}_best.pth"
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)
    
    metrics_csv = train_dir / "metrics.csv"
    with open(metrics_csv, "w") as f:
        f.write("epoch,train_loss,val_loss,val_mse,val_mae,val_rmse,lr,wall_time_s\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, global_step, best_val_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler, step_ckpt_path, args.save_freq, global_step,
            best_val_loss, logger,
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, logger)
        
        # Check if best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            }, best_ckpt_path)
            logger.info(f"Saved best model: val_loss={val_loss:.6f}")
        
        # Cleanup step checkpoint at epoch end
        if step_ckpt_path.exists() and not is_best:
            step_ckpt_path.unlink(missing_ok=True)
        
        wall_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        
        # Log metrics
        log_line = (
            f"{epoch},{train_loss:.6f},{val_loss:.6f},"
            f"{val_metrics['mse']:.6f},{val_metrics['mae']:.6f},{val_metrics['rmse']:.6f},"
            f"{lr:.2e},{wall_time:.1f}\n"
        )
        with open(metrics_csv, "a") as f:
            f.write(log_line)
        
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={lr:.2e} | "
            f"time={wall_time:.1f}s"
        )
    
    # Final test evaluation
    logger.info("=" * 60)
    logger.info("Final test evaluation")
    logger.info("=" * 60)
    
    # Load best model
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best model from {best_ckpt_path}")
    
    test_metrics = test_evaluate(model, test_loader, device, logger)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save test metrics
    test_metrics_path = args.out_dir / "infer" / "metrics" / f"{modality}_test_metrics.json"
    test_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_metrics_path, "w") as f:
        json.dump({
            "modality": modality,
            "model_size": model_size,
            "checkpoint": str(best_ckpt_path),
            "splits_file": str(splits_path),
            "metrics": test_metrics,
            "n_test": len(test_idx),
        }, f, indent=2)
    
    logger.info(f"Saved test metrics to {test_metrics_path}")
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best model: {best_ckpt_path}")
    logger.info(f"Metrics: {metrics_csv}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
