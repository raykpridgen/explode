#!/usr/bin/env python3
"""
Preprocess HEAT simulation data for MORPH surrogate modeling.

Converts raw NPZ timestep files into consolidated, packed NPZ ready for training.
Handles both CYL and PLI modalities with 9-channel packing per design.md.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# ==================== CONFIGURATION ====================

# 9-channel packing per design.md
CYL_CHANNEL_NAMES = [
    "Rcoord",
    "Zcoord",
    "vofm_maincharge",
    "vofm_booster",
    "vofm_wall",
    "vofm_Void",
    "av_density",
    "av_pressure",
    "speed",
]

PLI_CHANNEL_NAMES = [
    "Rcoord",
    "Zcoord",
    "vofm_maincharge",
    "vofm_striker",
    "vofm_case",
    "vofm_outside_air",
    "av_density",
    "av_pressure",
    "speed",
]

# Map design channel names to raw NPZ keys
CYL_RAW_KEY_MAP = {
    "Rcoord": "Rcoord",
    "Zcoord": "Zcoord",
    "vofm_maincharge": "vofm_maincharge",
    "vofm_booster": "vofm_booster",
    "vofm_wall": "vofm_wall",
    "vofm_Void": "vofm_Void",
    "av_density": "av_density",
    "av_pressure": "av_pressure",
    "speed": "__computed_hypot__",  # Special marker
}

PLI_RAW_KEY_MAP = {
    "Rcoord": "Rcoord",
    "Zcoord": "Zcoord",
    "vofm_maincharge": "vofm_maincharge",
    "vofm_striker": "vofm_striker",
    "vofm_case": "vofm_case",
    "vofm_outside_air": "vofm_outside_air",
    "av_density": "av_density",
    "av_pressure": "av_pressure",
    "speed": "__computed_hypot__",
}

# ==================== LOGGING ====================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the preprocessor."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# ==================== DATA LOADING ====================

def discover_instances(data_root: Path, modality: str, flat_structure: bool = False) -> List[Path]:
    """
    Discover all simulation instance directories or files.

    Args:
        data_root: Root directory containing raw data
        modality: 'cyl' or 'pli'
        flat_structure: If True, look for flat NPZ files instead of idXXXXX directories

    Returns:
        List of Paths to instance directories (idXXXXX) or modality dir for flat structure
    """
    modality_dir = data_root / modality
    if not modality_dir.exists():
        raise FileNotFoundError(f"Modality directory not found: {modality_dir}")

    # Check for flat structure (NPZ files directly in modality_dir)
    if flat_structure:
        logger.info(f"Using flat structure for {modality}")
        return [modality_dir]

    # Find all idXXXXX directories
    instance_dirs = []
    pattern = re.compile(r"^id\d{5}$")

    for item in modality_dir.iterdir():
        if item.is_dir() and pattern.match(item.name):
            instance_dirs.append(item)

    # If no instance dirs found, check for flat structure
    if not instance_dirs:
        npz_files = list(modality_dir.glob("*.npz"))
        if npz_files:
            logger.warning(f"No idXXXXX directories found, but found {len(npz_files)} NPZ files.")
            logger.warning(f"Use --flat-structure flag to process flat directory layout.")

    instance_dirs.sort()
    return instance_dirs


def get_instance_id_from_filename(filename: str, modality: str) -> Optional[str]:
    """
    Extract instance ID from NPZ filename.
    
    Args:
        filename: NPZ filename
        modality: 'cyl' or 'pli'
    
    Returns:
        Instance ID string (e.g., 'id00001') or None
    """
    if modality == "cyl":
        # Pattern: cx241203_id00001_pvi_idx00000.npz
        match = re.search(r"id(\d{5})", filename)
    elif modality == "pli":
        # Pattern: pli240420_id00001_pvi_idx00000.npz
        match = re.search(r"id(\d{5})", filename)
    else:
        return None
    
    if match:
        return f"id{match.group(1)}"
    return None


def load_instance_npz_files(instance_dir: Path, instance_id: Optional[str] = None, modality: Optional[str] = None) -> List[Path]:
    """
    Find and sort all NPZ timestep files for an instance.

    Args:
        instance_dir: Path to instance directory (e.g., id00001) or modality dir for flat structure
        instance_id: Instance ID for filtering in flat structure (e.g., 'id00001')
        modality: Modality for flat structure filename parsing

    Returns:
        Sorted list of NPZ file paths
    """
    if instance_id and modality:
        # Flat structure: filter by instance ID in filename
        all_npz_files = list(instance_dir.glob("*.npz"))
        npz_files = []
        for f in all_npz_files:
            file_instance_id = get_instance_id_from_filename(f.name, modality)
            if file_instance_id == instance_id:
                npz_files.append(f)
    else:
        # Hierarchical structure: all NPZ files in directory belong to this instance
        npz_files = list(instance_dir.glob("*.npz"))

    # Sort by index in filename (e.g., idx00000, idx00001, ...)
    def sort_key(p: Path) -> int:
        match = re.search(r"idx(\d+)", p.name)
        if match:
            return int(match.group(1))
        return 0

    npz_files.sort(key=sort_key)
    return npz_files


def discover_instances_flat(data_root: Path, modality: str) -> List[str]:
    """
    Discover all instance IDs from flat directory structure.
    
    Args:
        data_root: Root directory containing raw data
        modality: 'cyl' or 'pli'
    
    Returns:
        List of instance ID strings (e.g., ['id00001', 'id00002', ...])
    """
    modality_dir = data_root / modality
    if not modality_dir.exists():
        raise FileNotFoundError(f"Modality directory not found: {modality_dir}")

    # Find all unique instance IDs from NPZ filenames
    instance_ids = set()
    npz_files = list(modality_dir.glob("*.npz"))
    
    for npz_file in npz_files:
        instance_id = get_instance_id_from_filename(npz_file.name, modality)
        if instance_id:
            instance_ids.add(instance_id)
    
    sorted_ids = sorted(instance_ids)
    logger.info(f"Discovered {len(sorted_ids)} instances from flat directory")
    return sorted_ids


def load_npz_timestep(npz_path: Path, dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
    """
    Load a single timestep NPZ file.

    Args:
        npz_path: Path to NPZ file
        dtype: Target dtype for arrays

    Returns:
        Dictionary of arrays
    """
    data = np.load(npz_path)
    result = {}
    for key in data.keys():
        arr = data[key]
        # Handle scalar sim_time
        if arr.shape == ():
            result[key] = np.array(arr).astype(dtype)
        else:
            result[key] = arr.astype(dtype)
    return result


# ==================== CHANNEL PACKING ====================

def broadcast_coord(coord_1d: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Broadcast 1D coordinate array to target 2D shape.

    For CYL: Rcoord is (W,), Zcoord is (H,) -> broadcast to (H, W)
    For PLI: Rcoord is (W,), Zcoord is (H,) -> broadcast to (H, W)

    Args:
        coord_1d: 1D coordinate array
        target_shape: Target (H, W) shape

    Returns:
        Broadcasted 2D array
    """
    H, W = target_shape

    if coord_1d.shape[0] == W:
        # Rcoord: broadcast across rows (horizontal axis)
        return np.broadcast_to(coord_1d[None, :], (H, W))
    elif coord_1d.shape[0] == H:
        # Zcoord: broadcast across columns (vertical axis)
        return np.broadcast_to(coord_1d[:, None], (H, W))
    else:
        raise ValueError(
            f"Cannot broadcast coord of shape {coord_1d.shape} to {target_shape}"
        )


def compute_speed(u_vel: np.ndarray, w_vel: np.ndarray) -> np.ndarray:
    """
    Compute velocity magnitude: sqrt(U^2 + W^2).

    Args:
        u_vel: Uvelocity array
        w_vel: Wvelocity array

    Returns:
        Speed array (hypot)
    """
    return np.hypot(u_vel, w_vel)


def pack_cyl_channels(raw_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Pack CYL raw data into 9-channel format.

    Channels (per design.md):
        0: Rcoord (broadcast)
        1: Zcoord (broadcast)
        2: vofm_maincharge
        3: vofm_booster
        4: vofm_wall
        5: vofm_Void
        6: av_density
        7: av_pressure (log1p)
        8: speed (hypot(U, W))

    Args:
        raw_data: Dictionary of raw NPZ arrays

    Returns:
        (9, H, W) packed array
    """
    # Determine target shape from velocity field
    H, W = raw_data["Uvelocity"].shape

    # Initialize packed array
    packed = np.zeros((9, H, W), dtype=np.float32)

    # Channel 0: Rcoord (broadcast from 1D)
    packed[0] = broadcast_coord(raw_data["Rcoord"], (H, W))

    # Channel 1: Zcoord (broadcast from 1D)
    packed[1] = broadcast_coord(raw_data["Zcoord"], (H, W))

    # Channel 2: main charge
    packed[2] = raw_data["vofm_maincharge"]

    # Channel 3: booster
    packed[3] = raw_data["vofm_booster"]

    # Channel 4: wall
    packed[4] = raw_data["vofm_wall"]

    # Channel 5: void/air
    packed[5] = raw_data["vofm_Void"]

    # Channel 6: average density
    packed[6] = raw_data["av_density"]

    # Channel 7: average pressure (log1p for spike normalization)
    packed[7] = np.log1p(raw_data["av_pressure"])

    # Channel 8: speed (hypot of U and W velocities)
    packed[8] = compute_speed(raw_data["Uvelocity"], raw_data["Wvelocity"])

    return packed


def pack_pli_channels(raw_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Pack PLI raw data into 9-channel format.

    Channels (per design.md):
        0: Rcoord (broadcast)
        1: Zcoord (broadcast)
        2: vofm_maincharge
        3: vofm_striker
        4: vofm_case (wall)
        5: vofm_outside_air (void)
        6: av_density
        7: av_pressure (log1p)
        8: speed (hypot(U, W))

    Args:
        raw_data: Dictionary of raw NPZ arrays

    Returns:
        (9, H, W) packed array
    """
    # Determine target shape from velocity field
    H, W = raw_data["Uvelocity"].shape

    # Initialize packed array
    packed = np.zeros((9, H, W), dtype=np.float32)

    # Channel 0: Rcoord (broadcast from 1D)
    packed[0] = broadcast_coord(raw_data["Rcoord"], (H, W))

    # Channel 1: Zcoord (broadcast from 1D)
    packed[1] = broadcast_coord(raw_data["Zcoord"], (H, W))

    # Channel 2: main charge
    packed[2] = raw_data["vofm_maincharge"]

    # Channel 3: striker
    packed[3] = raw_data["vofm_striker"]

    # Channel 4: case (wall)
    packed[4] = raw_data["vofm_case"]

    # Channel 5: outside air (void)
    packed[5] = raw_data["vofm_outside_air"]

    # Channel 6: average density
    packed[6] = raw_data["av_density"]

    # Channel 7: average pressure (log1p for spike normalization)
    packed[7] = np.log1p(raw_data["av_pressure"])

    # Channel 8: speed (hypot of U and W velocities)
    packed[8] = compute_speed(raw_data["Uvelocity"], raw_data["Wvelocity"])

    return packed


# ==================== INSTANCE PROCESSING ====================

def process_instance(
    instance_dir: Path,
    modality: str,
    logger: logging.Logger,
    instance_id: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Process all timesteps for a single simulation instance.

    Args:
        instance_dir: Path to instance directory (or modality dir for flat structure)
        modality: 'cyl' or 'pli'
        logger: Logger instance
        instance_id: Instance ID for flat structure (e.g., 'id00001')

    Returns:
        Tuple of (volume, timesteps, r_grid, z_grid) or None if failed
        - volume: (T, 9, H, W) array
        - timesteps: (T,) array of sim_time values
        - r_grid: (H, W) broadcast R coordinates
        - z_grid: (H, W) broadcast Z coordinates
    """
    if instance_id:
        # Flat structure: instance_dir is actually the modality directory
        logger.debug(f"Processing instance: {instance_id} (flat structure)")
        npz_files = load_instance_npz_files(instance_dir, instance_id, modality)
    else:
        # Hierarchical structure: instance_dir is the instance directory
        instance_id = instance_dir.name
        logger.debug(f"Processing instance: {instance_id}")
        npz_files = load_instance_npz_files(instance_dir)
    
    if not npz_files:
        logger.warning(f"No NPZ files found for {instance_id}")
        return None

    logger.debug(f"Found {len(npz_files)} timesteps for {instance_id}")

    # Load first timestep to get shape info
    first_data = load_npz_timestep(npz_files[0])
    H, W = first_data["Uvelocity"].shape

    # Initialize storage
    n_timesteps = len(npz_files)
    volume = np.zeros((n_timesteps, 9, H, W), dtype=np.float32)
    sim_times = np.zeros(n_timesteps, dtype=np.float32)

    # Process each timestep
    for t, npz_path in enumerate(npz_files):
        try:
            raw_data = load_npz_timestep(npz_path)

            # Store sim_time
            sim_times[t] = raw_data.get("sim_time", 0.0)

            # Pack channels based on modality
            if modality == "cyl":
                volume[t] = pack_cyl_channels(raw_data)
            elif modality == "pli":
                volume[t] = pack_pli_channels(raw_data)
            else:
                raise ValueError(f"Unknown modality: {modality}")

        except Exception as e:
            logger.error(f"Error processing {npz_path}: {e}")
            return None

    # Store grid coordinates (broadcast from first timestep)
    if modality == "cyl":
        r_grid = broadcast_coord(first_data["Rcoord"], (H, W))
        z_grid = broadcast_coord(first_data["Zcoord"], (H, W))
    else:
        r_grid = broadcast_coord(first_data["Rcoord"], (H, W))
        z_grid = broadcast_coord(first_data["Zcoord"], (H, W))

    logger.debug(f"Instance {instance_id} complete: {n_timesteps} timesteps")
    return volume, sim_times, r_grid, z_grid


# ==================== MAIN PROCESSING ====================

def preprocess_modality(
    data_root: Path,
    output_dir: Path,
    modality: str,
    max_instances: Optional[int] = None,
    logger: logging.Logger = None,
    flat_structure: bool = False,
) -> Path:
    """
    Preprocess all instances for a single modality.

    Args:
        data_root: Root directory containing raw data
        output_dir: Directory for output NPZ
        modality: 'cyl' or 'pli'
        max_instances: Limit number of instances (for testing)
        logger: Logger instance
        flat_structure: If True, expect flat directory structure (NPZ files directly in modality dir)

    Returns:
        Path to output NPZ file
    """
    logger = logger or logging.getLogger(__name__)

    logger.info(f"Starting preprocessing for modality: {modality}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output dir: {output_dir}")
    
    # Detect flat structure if not specified
    modality_dir = data_root / modality
    if not flat_structure and modality_dir.exists():
        # Check if there are idXXXXX directories
        has_instance_dirs = any(d.is_dir() and re.match(r"^id\d{5}$", d.name) for d in modality_dir.iterdir())
        # Check if there are NPZ files directly in the directory
        has_flat_npz = len(list(modality_dir.glob("*.npz"))) > 0
        
        if not has_instance_dirs and has_flat_npz:
            logger.info(f"Detected flat directory structure for {modality}")
            flat_structure = True

    if flat_structure:
        # Flat structure: discover instance IDs from filenames
        instance_ids = discover_instances_flat(data_root, modality)
        logger.info(f"Discovered {len(instance_ids)} instances from flat structure")
        
        if max_instances:
            instance_ids = instance_ids[:max_instances]
            logger.info(f"Limited to {len(instance_ids)} instances for testing")
        
        # Process each instance
        all_volumes = []
        all_timesteps = []
        all_instance_ids = []
        first_result = None
        
        for instance_id in tqdm(instance_ids, desc=f"Processing {modality}"):
            result = process_instance(modality_dir, modality, logger, instance_id)
            if result is None:
                logger.warning(f"Skipping instance: {instance_id}")
                continue

            volume, timesteps, r_grid, z_grid = result

            if first_result is None:
                first_result = (r_grid, z_grid)

            all_volumes.append(volume)
            all_timesteps.append(timesteps)
            all_instance_ids.append(instance_id)
    else:
        # Hierarchical structure
        instance_dirs = discover_instances(data_root, modality)
        logger.info(f"Discovered {len(instance_dirs)} instances")

        if max_instances:
            instance_dirs = instance_dirs[:max_instances]
            logger.info(f"Limited to {len(instance_dirs)} instances for testing")

        # Process instances
        all_volumes = []
        all_timesteps = []
        all_instance_ids = []

        # Use first instance for grid shape (assumes consistent)
        first_result = None

        for instance_dir in tqdm(instance_dirs, desc=f"Processing {modality}"):
            result = process_instance(instance_dir, modality, logger)
            if result is None:
                logger.warning(f"Skipping instance: {instance_dir.name}")
                continue

            volume, timesteps, r_grid, z_grid = result

            if first_result is None:
                first_result = (r_grid, z_grid)

            all_volumes.append(volume)
            all_timesteps.append(timesteps)
            all_instance_ids.append(instance_dir.name)

    if not all_volumes:
        raise RuntimeError(f"No valid instances processed for {modality}")

    logger.info(f"Successfully processed {len(all_volumes)} instances")

    # Stack volumes - note: T may vary per instance, so we pad to max T
    max_t = max(v.shape[0] for v in all_volumes)
    H, W = all_volumes[0].shape[2], all_volumes[0].shape[3]

    n_instances = len(all_volumes)
    
    # Create padded arrays
    logger.info(f"Creating padded arrays: ({n_instances}, {max_t}, 9, {H}, {W})")
    
    padded_volume = np.zeros((n_instances, max_t, 9, H, W), dtype=np.float32)
    padded_timesteps = np.zeros((n_instances, max_t), dtype=np.float32)
    valid_mask = np.zeros((n_instances, max_t), dtype=bool)

    # Fill padded arrays and free source arrays to save memory
    logger.info("Filling padded arrays...")
    for i, (vol, ts) in enumerate(zip(all_volumes, all_timesteps)):
        t = vol.shape[0]
        padded_volume[i, :t] = vol
        padded_timesteps[i, :t] = ts
        valid_mask[i, :t] = True
        # Free source array to save memory
        all_volumes[i] = None
    
    # Clear the lists to free memory
    all_volumes.clear()
    all_timesteps.clear()
    import gc
    gc.collect()
    logger.info("Arrays padded and source data freed")

    # Get grid from first instance
    r_grid, z_grid = first_result

    # Determine channel names
    channel_names = CYL_CHANNEL_NAMES if modality == "cyl" else PLI_CHANNEL_NAMES

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save output
    # Use uncompressed savez to avoid memory spike from compression
    # Compression requires holding entire dataset in memory, which causes OOM for large datasets
    output_path = output_dir / f"{modality}_data.npz"
    
    logger.info(f"Saving to {output_path} (uncompressed to save memory)...")
    
    # Save in chunks to minimize memory usage
    # np.savez (uncompressed) streams data instead of buffering for compression
    np.savez(
        output_path,
        volume=padded_volume,
        timesteps=padded_timesteps,
        valid_mask=valid_mask,
        instance_ids=np.array(all_instance_ids),
        channel_names=np.array(channel_names),
        r_grid=r_grid,
        z_grid=z_grid,
    )
    
    # Log info without keeping arrays in memory
    logger.info(f"Saved: {output_path}")
    logger.info(f"  Shape: ({n_instances}, {max_t}, 9, {H}, {W})")
    logger.info(f"  Instances: {n_instances}")
    logger.info(f"  Max timesteps: {max_t}")
    logger.info(f"  Grid: {H}x{W}")
    logger.info(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")

    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess HEAT simulation data for MORPH training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preprocess CYL modality (hierarchical structure with idXXXXX directories)
    python preprocess.py --modality cyl --data-root ./data --output-dir ./processed

    # Preprocess PLI modality (flat structure with NPZ files directly in data/pli/)
    python preprocess.py --modality pli --data-root ./data --output-dir ./processed --flat-structure

    # Auto-detect structure (will detect flat if no idXXXXX directories found)
    python preprocess.py --modality pli --data-root ./data --output-dir ./processed

    # Process only first 10 instances for testing
    python preprocess.py --modality cyl --max-instances 10 --verbose

Directory structure support:
    Hierarchical (default for CYL):
        data/cyl/
            id00001/
                cx241203_id00001_pvi_idx00000.npz
                ...
            id00002/
                ...
    
    Flat (common for PLI downloads):
        data/pli/
            pli240420_id00001_pvi_idx00000.npz
            pli240420_id00001_pvi_idx00001.npz
            ...
            pli240420_id00002_pvi_idx00000.npz
            ...

Output format:
    Creates {modality}_data.npz containing:
    - volume: (N, T, 9, H, W) packed channels
    - timesteps: (N, T) simulation time values
    - valid_mask: (N, T) boolean mask for valid timesteps
    - instance_ids: (N,) instance directory names
    - channel_names: (9,) channel name strings
    - r_grid: (H, W) R coordinates
    - z_grid: (H, W) Z coordinates
""",
    )

    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["cyl", "pli"],
        help="Simulation modality to preprocess",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Root directory containing raw data (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./processed"),
        help="Output directory for processed NPZ (default: ./processed)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum number of instances to process (for testing)",
    )
    parser.add_argument(
        "--flat-structure",
        action="store_true",
        help="Use flat directory structure (NPZ files directly in modality dir, not in idXXXXX subdirs). "
             "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("HEAT Data Preprocessor")
    logger.info("=" * 60)

    # Validate paths
    if not args.data_root.exists():
        logger.error(f"Data root not found: {args.data_root}")
        sys.exit(1)

    try:
        output_path = preprocess_modality(
            data_root=args.data_root,
            output_dir=args.output_dir,
            modality=args.modality,
            max_instances=args.max_instances,
            logger=logger,
            flat_structure=args.flat_structure,
        )
        logger.info("=" * 60)
        logger.info(f"Preprocessing complete: {output_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
