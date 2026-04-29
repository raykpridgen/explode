#!/usr/bin/env python3
"""
Train or evaluate a MORPH surrogate on packed HEAT NPZ volumes.

See design.md (repo root) and specs/issues.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_morph_on_path(morph_root: Path) -> Path:
    morph_root = morph_root.resolve()
    s = str(morph_root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return morph_root


def _parse_extra_log1p(s: str | None) -> set[int]:
    if not s:
        return set()
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


class _XYDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _synth_demo_volume(modality: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    n, t, d, h, w = 2100, 24, 1, 64, 64
    k = 11 if modality == "cyl" else 14
    return rng.standard_normal((n, t, d, h, w, k)).astype(np.float32) * 0.1 + 0.5


def main() -> None:
    p = argparse.ArgumentParser(description="HEAT × MORPH surrogate training")
    p.add_argument("--modality", choices=["cyl", "pli"], required=True)
    p.add_argument("--data", type=str, default="", help="Path to HEAT .npz (ignored with --demo)")
    p.add_argument(
        "--morph-root",
        type=str,
        default=str(REPO_ROOT / "code" / "MORPH"),
        help="Path to MORPH repository root (contains src/)",
    )
    p.add_argument("--packing-variant", type=str, default="default")
    p.add_argument("--demo", action="store_true", help="Synthetic data smoke run (2100×24×1×64×64)")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--stats-dir", type=str, default=str(REPO_ROOT / "artifacts" / "revin_stats"))
    p.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "artifacts" / "heat_morph_runs"))
    p.add_argument("--ar-order", type=int, default=1)
    p.add_argument("--max-ar-order", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device-idx", type=int, default=0)
    p.add_argument("--model-size", choices=["Ti", "S", "M", "L"], default="Ti")
    p.add_argument("--heads-xa", type=int, default=32)
    p.add_argument("--download-model", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--ckpt-from", choices=["FM", "FT"], default="FM")
    p.add_argument("--ft-level4", action="store_true", default=True)
    p.add_argument("--no-ft-level4", action="store_false", dest="ft_level4")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-level4", type=float, default=1e-4)
    p.add_argument("--wd-level4", type=float, default=0.0)
    p.add_argument("--lr-scheduler", action="store_true")
    p.add_argument("--log1p-channels", type=str, default="", help="Comma-separated raw channel indices")
    p.add_argument("--metrics-max-batches", type=int, default=5)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to .pth (required for --eval-only; optional warm-start for training)",
    )
    args = p.parse_args()

    if args.eval_only and not args.resume:
        p.error("--eval-only requires --resume pointing to a saved checkpoint")

    morph_root = _ensure_morph_on_path(Path(args.morph_root))

    from heat_morph.channels import build_morph_volume, get_packing_spec
    from heat_morph.io import load_heat_npz
    from heat_morph.metrics import per_group_mse
    from heat_morph.normalize import revin_fit_apply
    from heat_morph.split import split_instance_indices, take_instances

    from src.utils.data_preparation_fast import FastARDataPreparer  # type: ignore
    from src.utils.device_manager import DeviceManager  # type: ignore
    from src.utils.metrics_3d import Metrics3DCalculator  # type: ignore
    from src.utils.select_fine_tuning_parameters import SelectFineTuningParameters  # type: ignore
    from src.utils.trainers import Trainer  # type: ignore
    from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression  # type: ignore

    spec = get_packing_spec(args.modality, args.packing_variant)
    extra_log1p = _parse_extra_log1p(args.log1p_channels)

    if args.demo:
        raw = _synth_demo_volume(args.modality)
    else:
        if not args.data:
            p.error("--data required unless --demo")
        raw, _ = load_heat_npz(Path(args.data))

    vol_uptf7 = build_morph_volume(raw, spec, extra_log1p_raw_indices=extra_log1p)
    del raw

    n_inst = vol_uptf7.shape[0]
    tr_idx, va_idx, te_idx = split_instance_indices(n_inst, seed=args.split_seed)
    if n_inst != 2100:
        print(f"warning: N={n_inst} != 2100; using proportional 80/10/10 split")

    splits_np = {
        "train": take_instances(vol_uptf7, tr_idx),
        "val": take_instances(vol_uptf7, va_idx),
        "test": take_instances(vol_uptf7, te_idx),
    }
    prefix = f"norm_{args.modality}"
    splits_norm = revin_fit_apply(morph_root, args.stats_dir, splits_np["train"], splits_np, prefix)

    preparer = FastARDataPreparer(ar_order=args.ar_order)

    def ar_split(name: str):
        arr = splits_norm[name]
        arr_dhwcf = arr.transpose(0, 1, 4, 5, 6, 3, 2)
        x, y = preparer.prepare(np.ascontiguousarray(arr_dhwcf))
        return x, y

    x_tr, y_tr = ar_split("train")
    x_va, y_va = ar_split("val")
    x_te, y_te = ar_split("test")

    devices = DeviceManager.list_devices()
    device = devices[args.device_idx] if devices else torch.device("cpu")

    MORPH_MODELS = {
        "Ti": [8, 256, 4, 4, 1024],
        "S": [8, 512, 8, 4, 2048],
        "M": [8, 768, 12, 8, 3072],
        "L": [8, 1024, 16, 16, 4096],
    }
    filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]
    dropout, emb_dropout = 0.1, 0.1

    model = ViT3DRegression(
        patch_size=8,
        dim=dim,
        depth=depth,
        heads=heads,
        heads_xa=args.heads_xa,
        mlp_dim=mlp_dim,
        max_components=3,
        conv_filter=filters,
        max_ar=args.max_ar_order,
        max_patches=4096,
        max_fields=3,
        dropout=dropout,
        emb_dropout=emb_dropout,
        lora_r_attn=0,
        lora_r_mlp=0,
        lora_alpha=None,
        lora_p=0.0,
        model_size=args.model_size,
    ).to(device)

    ft_args = SimpleNamespace(
        ft_level4=bool(args.ft_level4),
        ft_level1=False,
        ft_level2=False,
        ft_level3=False,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_level4=args.lr_level4,
        wd_level4=args.wd_level4,
        rank_lora_attn=0,
        rank_lora_mlp=0,
        lora_p=0.0,
    )
    selector = SelectFineTuningParameters(model, ft_args)
    optimizer = selector.configure_levels()

    criterion = nn.MSELoss()
    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    savepath_model = morph_root / "models"
    savepath_model.mkdir(parents=True, exist_ok=True)

    fname = None
    if args.download_model and args.ckpt_from == "FM" and args.checkpoint is None:
        from huggingface_hub import hf_hub_download  # type: ignore

        if args.model_size == "Ti":
            fname = "morph-Ti-FM-max_ar1_ep225.pth"
        elif args.model_size == "S":
            fname = "morph-S-FM-max_ar1_ep225.pth"
        elif args.model_size == "M":
            fname = "morph-M-FM-max_ar1_ep290_latestbatch.pth"
        elif args.model_size == "L":
            fname = "morph-L-FM-max_ar16_ep189_latestbatch.pth"
        weights_path = hf_hub_download(
            repo_id="mahindrautela/MORPH",
            filename=fname,
            subfolder="models/FM",
            repo_type="model",
            resume_download=True,
            local_dir=str(morph_root),
            local_dir_use_symlinks=False,
        )
        args.checkpoint = str(Path(weights_path).name)
        print(f"downloaded FM weights to {weights_path}")

    def load_weights(target: nn.Module) -> None:
        if args.resume:
            ckpt_path = Path(args.resume)
        elif args.ckpt_from == "FM" and args.checkpoint:
            ckpt_path = savepath_model / "FM" / args.checkpoint
        else:
            print("no checkpoint (--checkpoint / --resume / --download-model); training from init")
            return
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        print(f"loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if not isinstance(state, dict):
            raise TypeError("checkpoint must contain a state_dict or model_state_dict")
        if state and next(iter(state)).startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        strict = bool(args.ft_level4)
        missing, unexpected = target.load_state_dict(state, strict=strict)
        print("missing keys (subset):", list(missing)[:8], "count", len(missing))
        print("unexpected keys:", unexpected)

    load_weights(model)

    out_dir = Path(args.out_dir) / f"{args.modality}_{args.model_size}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "packing": str(spec)}, f, indent=2, default=str)

    train_loader = DataLoader(
        _XYDataset(x_tr, y_tr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        _XYDataset(x_va, y_va),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        _XYDataset(x_te, y_te),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    hist_path = out_dir / "metrics_history.jsonl"

    def log_group_metrics(split: str, loader: DataLoader, max_batches: int) -> dict:
        model.eval()
        agg: dict[str, float] = {}
        n = 0
        with torch.no_grad():
            for i, (xb, yb) in enumerate(loader):
                if i >= max_batches:
                    break
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                _, _, pred = model(xb)
                g = per_group_mse(pred, yb, spec)
                for k, v in g.items():
                    agg[k] = agg.get(k, 0.0) + v
                n += 1
        return {k: v / max(n, 1) for k, v in agg.items()}

    def extended_test_metrics() -> dict:
        model.eval()
        mse_tot = 0.0
        mae_tot = 0.0
        rmse_tot = 0.0
        ssim_tot = 0.0
        nb = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                _, _, pred = model(images)
                mse_tot += float(nn.functional.mse_loss(pred, targets).item())
                mae_tot += float((pred - targets).abs().mean().item())
                rmse_tot += float(Metrics3DCalculator.calculate_rmse(pred, targets).item())
                ssim_tot += float(Metrics3DCalculator.calculate_ssim(pred, targets).item())
                nb += 1
        if nb == 0:
            return {}
        return {
            "mse": mse_tot / nb,
            "mae": mae_tot / nb,
            "rmse": rmse_tot / nb,
            "ssim": ssim_tot / nb,
            "note": "computed in RevIN-normalized space",
        }

    if args.eval_only:
        metrics = extended_test_metrics()
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))
        return

    if args.resume and not args.eval_only:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("loaded optimizer from resume (warm start)")

    best_val = float("inf")
    best_path = out_dir / "checkpoint_best.pth"
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss = Trainer.train_singlestep(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            scheduler,
            str(out_dir / "model_stub"),
            False,
            1000,
        )
        va_loss = Trainer.validate_singlestep(model, val_loader, criterion, device)
        if args.lr_scheduler and scheduler is not None:
            scheduler.step(va_loss)

        groups = log_group_metrics("val", val_loader, args.metrics_max_batches)
        rec = {
            "epoch": epoch,
            "train_mse": tr_loss,
            "val_mse": va_loss,
            "val_groups": groups,
            "lr": optimizer.param_groups[0]["lr"],
            "wall_s": time.time() - t0,
        }
        with open(hist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va_loss,
                },
                best_path,
            )

    test_metrics = extended_test_metrics()
    with open(out_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print("saved", best_path)
    print("test", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
