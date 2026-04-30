# HEAT × MORPH Surrogate Modeling Pipeline

A production-ready pipeline for training physics-informed surrogate models on LANL HEAT simulation data using the MORPH neural operator architecture.

## Overview

This pipeline transforms raw HEAT simulation data (CYL and PLI modalities) into trained surrogate models capable of autoregressive prediction. It handles data download, preprocessing, normalization, training with checkpoint management, and comprehensive visualization.

**Key Features:**
- **HPC-ready**: Slurm-compatible scripts with modular design
- **Memory-efficient**: Streaming data loading, step-by-step rollout inference
- **Reproducible**: Fixed seeds, persistent splits, version-tracked checkpoints
- **9-channel packing**: Condenses HEAT's rich channel set into MORPH's `(F=3, C=3)` constraint
- **Full visualization**: Training curves, rollout GIFs, prediction diffs, static comparisons

## Quick Start

```bash
# 1. Download data (choose one method)

# Method A: Streaming download - one instance at a time (RECOMMENDED)
# Automatically resumes, no URL list files, checks each instance
python scripts/download_streaming.py --modality cyl --data-root ./data

# Method B: Batch selective download (generates URL lists)
# python scripts/download_selective.py --modality cyl --data-root ./data

# Method C: Full tar download (~50GB for CYL, ~2.6TB for PLI)
# ./scripts/download_data.sh -m cyl -o ./data

# 2. Preprocess raw data to packed NPZ
# For CYL (hierarchical structure with idXXXXX directories)
python scripts/preprocess.py --modality cyl --data-root ./data --output-dir ./processed

# For PLI with flat structure (all NPZ files directly in data/pli/)
python scripts/preprocess.py --modality pli --data-root ./data --output-dir ./processed --flat-structure

# Or let it auto-detect the structure
python scripts/preprocess.py --modality pli --data-root ./data --output-dir ./processed

# 3. Train surrogate model (Medium size, 5 epochs)
# For CYL (560x200 grid)
python scripts/train_explode.py \
    --modality cyl \
    --data ./processed/cyl_data.npz \
    --model-size M \
    --epochs 5 \
    --batch-size 8

# For PLI (1120x400 grid) - use smaller batch size
python scripts/train_explode.py \
    --modality pli \
    --data ./processed/pli_data.npz \
    --model-size M \
    --epochs 5 \
    --batch-size 2

# 4. Visualize training metrics and test rollouts
python scripts/vis.py \
    --modality cyl \
    --checkpoint out/train/cyl/models/cyl_best.pth
```

**Output structure:**
```
out/
    splits.json                    # Train/val/test instance IDs
    train/cyl/
        metrics.csv                # Per-epoch training log
        models/cyl_best.pth        # Best checkpoint
    infer/plots/cyl/
        metrics_*.png              # Training curves
        rollout_idXXXXX/           # Per-instance visualizations
            *_comparison.png      # Static frame grid
            *_predicted.gif       # Rollout animation
            *_actual.gif          # Ground truth
            *_diff.gif            # Prediction error
```

---

## Detailed Workflow

### Step 1: Data Download (`download_data.sh`)

Downloads HEAT tar archives from LANL Oceans11 and extracts simulation instances.

**URLs:**
- CYL: `https://oceans11.lanl.gov/heat/cyl/cx241203_fp16_full.tar` (~50 GB)
- PLI: `https://oceans11.lanl.gov/heat/pli/pli240420.tar` (~100 GB)

**Usage:**
```bash
# Auto-detect best download tool (aria2c > wget > curl)
./scripts/download_data.sh -m both -o ./data

# Use aria2c with 8 parallel connections (faster)
./scripts/download_data.sh -m cyl -t aria2c -c 8

# HPC (Slurm) - edit script to uncomment SBATCH directives, then:
sbatch ./scripts/download_data.sh -m cyl
```

**What it does:**
1. Downloads tar file to `{DATA_ROOT}/`
2. Extracts `id00001` through `id02100` folders
3. Each folder contains ~50-100 NPZ timestep files

**Expected duration:**
- Download: 30-90 minutes (depends on connection)
- Extraction: 10-30 minutes (depends on storage speed)

---

### Step 1 (Primary): Streaming Instance Download (`download_streaming.py`) ⭐ RECOMMENDED

**Best for reliable resume and checkpointing.** Downloads one instance at a time with per-instance completion checking. No URL list files - completely streaming.

**Storage comparison:**
- CYL (2100 instances): ~150GB
- PLI (2100 instances): ~300GB

**Resume strategy:**
- Hidden checkpoint file (`.cyl_download_checkpoint`) tracks last completed instance
- Each instance verified individually - if all files present, skip to next
- Partial instances are re-downloaded cleanly
- Run same command to resume from where you left off

**Usage:**
```bash
# Download all 2100 instances (auto-resumes if interrupted)
python scripts/download_streaming.py --modality cyl --data-root ./data

# Download specific range
python scripts/download_streaming.py --modality pli --start-id 1 --end-id 500

# Force restart from beginning (ignore checkpoint)
python scripts/download_streaming.py --modality cyl --restart

# More parallel connections per file
python scripts/download_streaming.py --modality cyl --connections 8
```

**What it does:**
1. Reads checkpoint file to find last completed instance (if resuming)
2. For each instance from checkpoint+1 to end:
   - Scan server directory for NPZ files
   - Check if all files already exist locally (complete)
   - If complete: skip, update checkpoint
   - If incomplete: clear partial files, download all via aria2c, verify, update checkpoint
3. On success: remove checkpoint file
4. On interrupt: checkpoint saved, resume on next run

**Key advantages over batch method:**
- **No URL list files** - doesn't create large intermediate files
- **Granular resume** - knows exactly which instance to continue from
- **Clean partial handling** - re-downloads partial instances completely (safer than resuming individual files)
- **Memory efficient** - only holds one instance's URLs in memory at a time

**Scaling estimates:**

| Resource | CYL (2100 inst) | PLI (2100 inst) |
|----------|----------------|-----------------|
| Wall time | ~2-4 hours | ~4-8 hours |
| Memory | ~2 GB | ~2 GB |
| Network | ~150 GB | ~300 GB |
| Output | ~150 GB | ~300 GB |
| CPU | 2-4 cores | 2-4 cores |

**HPC resource specification:**
```bash
# Slurm for streaming download (CYL)
#SBATCH --job-name=heat_stream
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=shared

python scripts/download_streaming.py --modality cyl --data-root /scratch/$USER/data
```

**Checkpoint file location:**
```
data/
  ├── .cyl_download_checkpoint    # Hidden file, last completed ID
  ├── cyl/
  │   ├── id00001/
  │   ├── id00002/
  │   └── ...
```

---

### Step 1 (Alternative): Selective Instance Download (`download_selective.py`)

**Recommended for limited storage/bandwidth.** Downloads individual simulation instances (id00001-id02100) rather than full tar archives.

**Storage comparison:**
- Selective (2100 instances): ~150GB for CYL, ~300GB for PLI
- Full tar: ~50GB for CYL, ~2.6TB for PLI

**URLs:**
- CYL: `https://oceans11.lanl.gov/heat/cyl/cx241203_fp16_full/idXXXXX/`
- PLI: `https://oceans11.lanl.gov/heat/pli/pli240420/idXXXXX/`

**Usage:**
```bash
# Download all 2100 instances for CYL
python scripts/download_selective.py --modality cyl --data-root ./data

# Download specific range (for parallel batch jobs)
python scripts/download_selective.py --modality pli --start-id 1 --end-id 500 --data-root ./data_part1

# Resume interrupted download (automatic)
python scripts/download_selective.py --modality cyl --data-root ./data

# Dry run to see what would be downloaded
python scripts/download_selective.py --modality cyl --dry-run --data-root ./data

# Fast mode with more connections
python scripts/download_selective.py --modality cyl --connections 8 --max-concurrent 32
```

**What it does:**
1. Scans Apache directory listings for each instance (id00001 to id02100)
2. Generates aria2c URL list with per-file output paths
3. Checks for existing files to support resume
4. Runs aria2c with parallel connections for efficient download
5. Verifies completeness after download

**Features:**
- **Resume support**: aria2c automatically resumes interrupted downloads
- **Existing file check**: Skips files that already exist (matches size/time)
- **Parallel downloads**: Configurable concurrent downloads (default: 16)
- **Per-instance directories**: Organizes files as `data/{cyl,pli}/idXXXXX/*.npz`

**Scaling estimates:**

| Resource | CYL (2100 inst) | PLI (2100 inst) |
|----------|----------------|-----------------|
| Wall time | ~2-4 hours | ~4-8 hours |
| Memory | ~2 GB | ~2 GB |
| Network | ~150 GB | ~300 GB |
| Output | ~150 GB | ~300 GB |
| CPU | 2-4 cores | 2-4 cores |

**HPC resource specification:**
```bash
# Slurm for selective download (CYL)
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard  # No GPU needed
```

**Parallel batch strategy:**
For very large downloads, split across multiple jobs:
```bash
# Job 1: Instances 1-700
python scripts/download_selective.py --modality pli --start-id 1 --end-id 700 &

# Job 2: Instances 701-1400
python scripts/download_selective.py --modality pli --start-id 701 --end-id 1400 &

# Job 3: Instances 1401-2100
python scripts/download_selective.py --modality pli --start-id 1401 --end-id 2100 &
```

---

### Step 2: Preprocessing (`preprocess.py`)

Converts raw timestep NPZ files into consolidated, channel-packed NPZ ready for training.

**Channel packing (9 channels → 3 fields × 3 components):**
| Slot | CYL | PLI | Transform |
|------|-----|-----|-----------|
| 0 | Rcoord | Rcoord | Broadcast from 1D |
| 1 | Zcoord | Zcoord | Broadcast from 1D |
| 2 | vofm_maincharge | vofm_maincharge | - |
| 3 | vofm_booster | vofm_striker | - |
| 4 | vofm_wall | vofm_case | - |
| 5 | vofm_Void | vofm_outside_air | - |
| 6 | av_density | av_density | - |
| 7 | av_pressure | av_pressure | `log1p(x)` |
| 8 | speed | speed | `hypot(U, W)` |

**Directory structure support:**

The script auto-detects directory structure, or you can specify with `--flat-structure`:

**Hierarchical (default for CYL):**
```
data/cyl/
    id00001/
        cx241203_id00001_pvi_idx00000.npz
        cx241203_id00001_pvi_idx00001.npz
        ...
    id00002/
        ...
```

**Flat (common for PLI downloads):**
```
data/pli/
    pli240420_id00001_pvi_idx00000.npz
    pli240420_id00001_pvi_idx00001.npz
    ...
    pli240420_id00002_pvi_idx00000.npz
    ...
```

**Usage:**
```bash
# Full preprocessing (2100 instances) - auto-detects structure
python scripts/preprocess.py --modality cyl --data-root ./data --output-dir ./processed

# Force flat structure mode (for PLI with flat directory)
python scripts/preprocess.py --modality pli --data-root ./data --output-dir ./processed --flat-structure

# Quick test (first 10 instances)
python scripts/preprocess.py --modality cyl --data-root ./data --output-dir ./processed_test --max-instances 10
```

**What it does:**
1. Detects directory structure (hierarchical `idXXXXX/` or flat)
2. Groups NPZ files by instance ID (from filename for flat structure)
3. Loads each timestep NPZ, extracts features
4. Applies `log1p` to pressure, `hypot(U,W)` to velocities
5. Broadcasts 1D R/Z coordinates to 2D grid
6. Pads variable-length trajectories to max T
7. Frees source arrays during padding to save memory
8. Saves `{modality}_data.npz` with fields (uncompressed to avoid OOM):
   - `volume`: (N, T, 9, H, W) float32
   - `timesteps`: (N, T) float32
   - `valid_mask`: (N, T) bool
   - `instance_ids`: (N,) str
   - `r_grid`, `z_grid`: (H, W) float32

**Scaling estimates:**

| Resource | CYL (2100 inst) | PLI (2100 inst) |
|----------|----------------|-----------------|
| Wall time | ~20-30 min | ~40-60 min |
| Memory | ~8 GB | ~16-24 GB |
| CPU cores | 4-8 | 4-8 |
| Output size | ~12 GB (uncompressed) | ~48 GB (uncompressed) |

**Memory efficiency note:**
The script uses **uncompressed NPZ format** (`np.savez` instead of `np.savez_compressed`) because compression requires holding the entire dataset in memory twice (once for data, once for compressed buffer). This would cause OOM for large datasets. The output files are larger but the preprocessing completes without memory issues.

**HPC resource specification:**
```bash
# Slurm for CYL preprocessing
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# For PLI (larger memory needs)
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
```

---

### Step 3: Training (`train_explode.py`)

Trains MORPH surrogate with autoregressive single-step prediction.

**Architecture:**
- Base: MORPH ViT3DRegression (Medium: 768-dim, 8-layer, 12-head)
- Input: `(F=3, C=3, D=1, H, W)` - packed channels as 3×3 field-component layout
- Output: Same shape at `t+1`
- Normalization: Per-instance normalization (computed on-the-fly in DataLoader)
- Schedule: Warmup + cosine decay (10% warmup, 5 epochs default)

**Memory efficiency:**
The training script uses **streaming DataLoaders** with on-the-fly normalization to handle large datasets:
- No materialization of all AR windows (~340GB for full CYL dataset)
- Per-instance normalization computed dynamically (not full RevIN on arrays)
- Peak memory: ~15-25GB depending on grid size and batch size
- For large grids (PLI 1120x400), use `--batch-size 2` or `--batch-size 4`

**Usage:**
```bash
# From scratch (Medium model, 5 epochs)
python scripts/train_explode.py \
    --modality cyl \
    --data ./processed/cyl_data.npz \
    --model-size M \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-4

# Resume from checkpoint
python scripts/train_explode.py \
    --modality cyl \
    --data ./processed/cyl_data.npz \
    --resume out/train/cyl/models/cyl_best.pth \
    --epochs 10  # Continue to epoch 10

# Evaluation only
python scripts/train_explode.py \
    --modality cyl \
    --data ./processed/cyl_data.npz \
    --eval-only \
    --resume out/train/cyl/models/cyl_best.pth
```

**What it does:**
1. Loads processed NPZ
2. Splits 80/10/10 (train/val/test) by instance ID
3. Saves `splits.json` for downstream use
4. Creates streaming DataLoaders with per-instance normalization on-the-fly
5. Generates autoregressive windows (t → t+1) dynamically during training
6. Trains with MSE loss, warmup+cosine LR
7. Saves step checkpoints (every N steps, cleaned at epoch end)
8. Keeps only best model by validation loss
9. Runs final test evaluation (MSE, MAE, RMSE, SSIM)

**Checkpointing strategy:**
- Step checkpoints: Saved every `--save-freq` steps (default: 100), deleted at epoch end
- Best model: `cyl_best.pth` kept permanently, overwritten when beaten
- Resume: Loads model + optimizer state, continues from saved epoch

**Scaling estimates:**

| Configuration | Wall time/epoch | Memory | GPU memory |
|-------------|-----------------|--------|------------|
| CYL, Ti, BS=4 | ~5 min | 4 GB | 6 GB |
| CYL, S, BS=4 | ~10 min | 8 GB | 10 GB |
| CYL, M, BS=4 | ~20 min | 16 GB | 16 GB |
| CYL, M, BS=8 | ~15 min | 24 GB | 20 GB |
| PLI, M, BS=2 | ~40 min | 15-20 GB | 20 GB |
| PLI, M, BS=4 | ~35 min | 20-25 GB | 20 GB |

**Full training (5 epochs, Medium model):**
- CYL: ~1.5-2 hours on single A100
- PLI: ~3-4 hours on single A100

**HPC resource specification:**
```bash
# Slurm for Medium model training
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
```

**Multi-GPU note:** The script uses single GPU by default. For multi-GPU, wrap model in `nn.DataParallel` or use the MORPH trainer classes directly.

---

### Step 4: Visualization (`vis.py`)

Comprehensive visualization of training metrics and test rollouts.

**Usage:**
```bash
# Full visualization (metrics + rollouts for test set)
python scripts/vis.py \
    --modality cyl \
    --checkpoint out/train/cyl/models/cyl_best.pth \
    --max-rollout 50 \
    --max-test-instances 5

# Metrics only (fast)
python scripts/vis.py --modality cyl --checkpoint out/train/cyl/models/cyl_best.pth --metrics-only

# Limited rollout for quick preview
python scripts/vis.py \
    --modality cyl \
    --checkpoint out/train/cyl/models/cyl_best.pth \
    --max-rollout 20 \
    --fps 10 \
    --max-test-instances 2
```

**What it does:**
1. **Training metrics** (from `metrics.csv`):
   - Loss curves (train vs val)
   - Validation metrics (MSE, MAE, RMSE)
   - Learning rate schedule
2. **Rollout inference** (per test instance):
   - Load best model
   - Run autoregressive rollout from initial frame
   - Denormalize predictions
3. **Visualizations** (per force channel: av_density, av_pressure, speed):
   - **Static comparison grid**: 8-frame subset, predicted vs actual side-by-side
   - **Predicted GIF**: Full rollout animation
   - **Actual GIF**: Ground truth animation
   - **Diff GIF**: Diverging colormap (RdBu_r) showing prediction error

**Output files per instance:**
```
rollout_idXXXXX/
    av_density_comparison.png      # Static grid: 8 frames
    av_density_predicted.gif       # Full rollout animation
    av_density_actual.gif          # Ground truth
    av_density_diff.gif            # Error visualization
    av_pressure_comparison.png
    av_pressure_predicted.gif
    ...
```

**Scaling estimates:**

| Task | Wall time | Memory | GPU |
|------|-----------|--------|-----|
| Metrics only | <10 sec | 2 GB | No |
| 1 rollout (50 steps, 1 instance) | ~2 min | 8 GB | Yes (8 GB) |
| 5 rollouts (50 steps, 5 instances) | ~10 min | 8 GB | Yes (8 GB) |
| Full test set (210 instances) | ~4 hours | 8 GB | Yes (8 GB) |

**HPC resource specification:**
```bash
# Slurm for visualization (GPU required for rollout)
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
```

---

## Complete Pipeline Example

```bash
#!/bin/bash
#SBATCH --job-name=heat_pipeline
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Set paths
export DATA_ROOT=./data
export PROCESSED=./processed
export OUT=./out

# 1. Download (skip if already present, auto-resumes if interrupted)
if [ ! -d "$DATA_ROOT/cyl/id00001" ]; then
    echo "=== Downloading CYL data (streaming method) ==="
    python scripts/download_streaming.py --modality cyl --data-root $DATA_ROOT
fi

# 2. Preprocess (skip if already present)
if [ ! -f "$PROCESSED/cyl_data.npz" ]; then
    echo "=== Preprocessing CYL data ==="
    python scripts/preprocess.py \
        --modality cyl \
        --data-root $DATA_ROOT \
        --output-dir $PROCESSED
fi

# 3. Train
echo "=== Training surrogate ==="
python scripts/train_explode.py \
    --modality cyl \
    --data $PROCESSED/cyl_data.npz \
    --model-size M \
    --epochs 5 \
    --batch-size 8 \
    --out-dir $OUT

# 4. Visualize
echo "=== Generating visualizations ==="
python scripts/vis.py \
    --modality cyl \
    --checkpoint $OUT/train/cyl/models/cyl_best.pth \
    --max-test-instances 5 \
    --max-rollout 50 \
    --fps 5

echo "=== Pipeline complete ==="
```

---

## Directory Structure

```
explode/
├── scripts/
│   ├── download_data.sh         # Full tar download script
│   ├── download_streaming.py    # Streaming instance downloader (RECOMMENDED)
│   ├── download_selective.py    # Batch selective downloader
│   ├── preprocess.py            # Raw → packed NPZ
│   ├── train_explode.py         # Training + evaluation
│   └── vis.py                   # Visualization
├── code/MORPH/                  # MORPH repository (imported)
├── data/                        # Raw HEAT data (created by download)
│   ├── cyl/
│   │   ├── cyl.csv
│   │   ├── id00001/
│   │   │   └── *.npz
│   │   └── ...
│   └── pli/
├── processed/                   # Preprocessed data (created by preprocess)
│   ├── cyl_data.npz
│   └── pli_data.npz
├── out/                         # All outputs (created by train/vis)
│   ├── splits.json
│   ├── train/
│   │   ├── cyl/
│   │   │   ├── metrics.csv
│   │   │   ├── train.log
│   │   │   └── models/
│   │   │       └── cyl_best.pth
│   │   └── pli/
│   ├── infer/
│   │   └── plots/
│   │       ├── cyl/
│   │       │   ├── metrics_*.png
│   │       │   └── rollout_idXXXXX/
│   │       │       └── *.gif, *.png
│   │       └── pli/
│   └── revin_stats/             # RevIN statistics cache
├── specs/
│   ├── plan.md                  # Project requirements
│   ├── design.md                # Design contracts
│   └── issues.md                # Design decisions log
└── README.md                    # This file
```

---

## Resource Planning Guide

Use this table to estimate resources for your HPC allocation:

| Stage | Input size | Output size | Min time | Recommended time | Min memory | Recommended memory | GPU |
|-------|-----------|-------------|----------|----------------|------------|------------------|-----|
| **Download Methods** ||||||||
| Streaming CYL (2100 inst) | - | ~150 GB | 2 hours | 4 hours | 2 GB | 8 GB | No |
| Streaming PLI (2100 inst) | - | ~300 GB | 4 hours | 8 hours | 2 GB | 16 GB | No |
| Batch CYL (2100 inst) | - | ~150 GB | 2 hours | 4 hours | 4 GB | 8 GB | No |
| Batch PLI (2100 inst) | - | ~300 GB | 4 hours | 8 hours | 8 GB | 16 GB | No |
| Full tar CYL | - | ~50 GB | 30 min | 1 hour | 4 GB | 8 GB | No |
| Full tar PLI | - | ~2.6 TB | 2 hours | 4 hours | 8 GB | 32 GB | No |
| **Processing** ||||||||
| Preprocess CYL | ~150 GB | ~6 GB | 20 min | 1 hour | 8 GB | 16 GB | No |
| Preprocess PLI | ~300 GB | ~24 GB | 40 min | 2 hours | 16 GB | 32 GB | No |
| Train (Ti, 5ep) | ~6 GB | ~2 GB | 30 min | 1 hour | 8 GB | 16 GB | Yes (8 GB) |
| Train (M, 5ep) | ~6 GB | ~2 GB | 2 hours | 4 hours | 16 GB | 32 GB | Yes (16 GB) |
| Vis (5 rollouts) | ~2 GB | ~1 GB | 10 min | 30 min | 8 GB | 16 GB | Yes (8 GB) |

**Typical full pipeline (CYL, Medium model, selective download):**
- **Total time**: ~8-10 hours wall clock (4h download + 0.5h preprocess + 2h train + 0.5h vis)
- **Storage**: ~160 GB (150GB raw + 6GB processed + 4GB outputs)
- **GPU**: 1× A100 (40 GB) or V100 (16 GB)
- **Memory**: 32 GB RAM

**Typical full pipeline (CYL, Medium model, full tar download):**
- **Total time**: ~4-6 hours wall clock (1h download + 0.5h preprocess + 2h train + 0.5h vis)
- **Storage**: ~60 GB (50GB raw + 6GB processed + 4GB outputs)
- **GPU**: 1× A100 (40 GB) or V100 (16 GB)
- **Memory**: 32 GB RAM
- **Note**: Requires 50GB temporary for tar download

**Full dataset (CYL + PLI, Medium models, selective download):**
- **Total time**: ~16-24 hours wall clock
- **Storage**: ~470 GB (450GB raw + 30GB processed + 8GB outputs)
- **GPU**: 2× jobs or sequential on 1× GPU
- **Memory**: 64 GB RAM

**Full dataset (CYL + PLI, Medium models, full tar download):**
- **Total time**: ~12-18 hours wall clock
- **Storage**: ~2.7 TB (2.65TB raw + 30GB processed + 8GB outputs)
- **Note**: PLI tar is 2.6TB - requires significant temporary storage

---

## Troubleshooting

### CUDA out of memory during training
```bash
# Reduce batch size
python scripts/train_explode.py ... --batch-size 2

# Use smaller model
python scripts/train_explode.py ... --model-size S
```

### Preprocessing killed (OOM)
```bash
# Process fewer instances at once
python scripts/preprocess.py ... --max-instances 1000

# Run sequentially for full dataset
for i in {1..21}; do
    start=$(( (i-1)*100 + 1 ))
    end=$(( i*100 ))
    # Preprocess subset... (modify script for range support)
done
```

### Slow training
```bash
# Increase batch size if memory allows
python scripts/train_explode.py ... --batch-size 16

# Use DataParallel (requires modifying script)
model = nn.DataParallel(model)
```

### RevIN shape mismatch
This occurs when N (instances) doesn't match between train/val/test. The script handles this by computing stats per split. Ensure you're using consistent `split_seed` across runs.

### Missing MORPH imports
Ensure MORPH is cloned in `code/MORPH/`:
```bash
git clone git@github.com:raykpridgen/MORPH.git code/MORPH
```

---

## Design Documents

- `specs/plan.md` - Original requirements and feature specifications
- `specs/design.md` - Detailed design contracts (channel packing, normalization, checkpointing)
- `specs/issues.md` - Design decision log with resolved questions

---

## Citation

If using this pipeline in research, please cite:
- MORPH: [HuggingFace MORPH](https://huggingface.co/mahindrautela/MORPH)
- HEAT Dataset: [LANL Oceans11 HEAT](https://oceans11.lanl.gov/heat/)

---

## License

See individual component licenses:
- MORPH: Check `code/MORPH/LICENSE`
- Pipeline scripts: MIT (add license file)
