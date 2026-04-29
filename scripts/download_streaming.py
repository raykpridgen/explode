#!/usr/bin/env python3
"""
Streaming instance downloader for HEAT dataset.

Downloads one instance at a time, checking completion as it goes.
No URL list files - downloads on-the-fly with per-instance checkpointing.

Resume strategy:
- Checkpoint file tracks last completed instance
- Each instance checked individually - if all NPZ files exist, skip
- Partial instances are re-downloaded (safer than trying to resume individual files)
"""

import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests
from tqdm import tqdm

# Base URLs
BASE_URLS = {
    "cyl": "https://oceans11.lanl.gov/heat/cyl/cx241203_fp16_full/",
    "pli": "https://oceans11.lanl.gov/heat/pli/pli240420/",
}

# File patterns
FILE_PATTERNS = {
    "cyl": r"cx241203_id\d+_pvi_idx\d+\.npz",
    "pli": r"pli240420_id\d+_pvi_idx\d+\.npz",
}


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_checkpoint_file(data_root: Path, modality: str) -> Path:
    """Get path to checkpoint file tracking last completed instance."""
    return data_root / f".{modality}_download_checkpoint"


def load_checkpoint(checkpoint_file: Path) -> int:
    """Load last completed instance ID from checkpoint."""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file) as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return 0
    return 0


def save_checkpoint(checkpoint_file: Path, instance_id: int) -> None:
    """Save last completed instance ID to checkpoint."""
    with open(checkpoint_file, "w") as f:
        f.write(str(instance_id))


def fetch_directory_listing(url: str, logger: logging.Logger) -> List[str]:
    """Fetch Apache directory listing and extract file links."""
    logger.debug(f"Fetching directory: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return []
    
    html = response.text
    pattern = r'href="([^"]+)"'
    links = re.findall(pattern, html)
    
    files = []
    for link in links:
        if link in ("../", "?C=N;O=D", "?C=M;O=A", "?C=S;O=A", "?C=D;O=A"):
            continue
        if link.endswith("/"):
            continue
        files.append(link)
    
    return files


def get_instance_files(modality: str, instance_id: int, logger: logging.Logger) -> List[str]:
    """Get list of NPZ files for a specific instance."""
    base_url = BASE_URLS[modality]
    instance_str = f"id{instance_id:05d}"
    instance_url = urljoin(base_url, f"{instance_str}/")
    
    files = fetch_directory_listing(instance_url, logger)
    pattern = re.compile(FILE_PATTERNS[modality])
    npz_files = [f for f in files if pattern.match(f)]
    
    return npz_files


def is_instance_complete(instance_dir: Path, expected_files: List[str]) -> bool:
    """
    Check if an instance is completely downloaded.
    
    Returns True only if:
    - Directory exists
    - All expected NPZ files exist
    - All files have size > 0
    - No .aria2 partial files present
    """
    if not instance_dir.exists():
        return False
    
    for filename in expected_files:
        file_path = instance_dir / filename
        
        # Check file exists and has content
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False
        
        # Check no partial download marker
        if (file_path.parent / (filename + ".aria2")).exists():
            return False
    
    return True


def clear_partial_files(instance_dir: Path, logger: logging.Logger) -> None:
    """Remove partial download artifacts before re-downloading."""
    if not instance_dir.exists():
        return
    
    # Remove .aria2 control files and partial .npz files
    for f in instance_dir.glob("*.aria2"):
        try:
            f.unlink()
            logger.debug(f"Removed partial marker: {f}")
        except OSError:
            pass
    
    # Remove zero-size or corrupted npz files
    for f in instance_dir.glob("*.npz"):
        if f.stat().st_size == 0:
            try:
                f.unlink()
                logger.debug(f"Removed empty file: {f}")
            except OSError:
                pass


def download_instance(
    modality: str,
    instance_id: int,
    data_root: Path,
    connections: int,
    max_retries: int,
    logger: logging.Logger,
) -> bool:
    """
    Download a single complete instance.
    
    Returns True if successful.
    """
    base_url = BASE_URLS[modality]
    instance_str = f"id{instance_id:05d}"
    instance_dir = data_root / modality / instance_str
    
    # Get list of files to download
    files = get_instance_files(modality, instance_id, logger)
    if not files:
        logger.error(f"No files found for {instance_str}")
        return False
    
    logger.info(f"Instance {instance_str}: {len(files)} files to download")
    
    # Check if already complete
    if is_instance_complete(instance_dir, files):
        logger.debug(f"Instance {instance_str} already complete, skipping")
        return True
    
    # Clear any partial artifacts
    clear_partial_files(instance_dir, logger)
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    # Build aria2c input
    aria2_input = []
    for filename in files:
        file_url = urljoin(base_url, f"{instance_str}/{filename}")
        out_path = str(instance_dir / filename)
        aria2_input.append(f"{file_url}\n  out={out_path}\n")
    
    # Run aria2c for this instance
    cmd = [
        "aria2c",
        "--input-file", "-",  # Read from stdin
        "--dir", str(data_root),
        "--max-connection-per-server", str(connections),
        "--split", str(connections),
        "--max-concurrent-downloads", str(min(16, len(files))),
        "--continue", "true",
        "--auto-resume", "true",
        "--remote-time", "true",
        "--allow-overwrite", "true",  # Overwrite partial files
        "--max-tries", str(max_retries),
        "--retry-wait", "5",
        "--timeout", "60",
        "--log-level", "warn",
        "--console-log-level", "warn",
        "--summary-interval", "0",  # Less verbose for single instance
    ]
    
    try:
        result = subprocess.run(
            cmd,
            input="".join(aria2_input),
            text=True,
            capture_output=True,
            timeout=3600,  # 1 hour timeout per instance
        )
        
        if result.returncode != 0:
            logger.error(f"aria2c failed for {instance_str}: {result.stderr}")
            return False
        
        # Verify completion
        if not is_instance_complete(instance_dir, files):
            logger.error(f"Instance {instance_str} incomplete after download")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout downloading {instance_str}")
        return False
    except Exception as e:
        logger.error(f"Error downloading {instance_str}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Streaming HEAT instance downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all 2100 CYL instances (resumes automatically)
    python scripts/download_streaming.py --modality cyl --data-root ./data

    # Download specific range
    python scripts/download_streaming.py --modality pli --start-id 1 --end-id 500

    # Fast mode with more connections
    python scripts/download_streaming.py --modality cyl --connections 8

    # Force restart from beginning (ignore checkpoint)
    python scripts/download_streaming.py --modality cyl --restart

Resume behavior:
    - Creates hidden checkpoint file: .{modality}_download_checkpoint
    - On restart, reads checkpoint and continues from next instance
    - Each instance checked individually - if complete, skip
    - If interrupted, re-run same command to continue
""",
    )
    
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["cyl", "pli"],
        help="Simulation modality to download",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Root directory for downloaded data (default: ./data)",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Starting instance ID (default: 1, or resume from checkpoint)",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=2100,
        help="Ending instance ID (default: 2100)",
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=4,
        help="Connections per server for aria2c (default: 4)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per file (default: 5)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore checkpoint and start from --start-id",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("=" * 60)
    logger.info("HEAT Streaming Instance Downloader")
    logger.info("=" * 60)
    logger.info(f"Modality: {args.modality}")
    logger.info(f"Data root: {args.data_root}")
    
    # Setup checkpoint
    checkpoint_file = get_checkpoint_file(args.data_root, args.modality)
    
    # Determine start ID
    if args.restart:
        start_id = args.start_id
        logger.info(f"Restart mode: starting from {start_id}")
    else:
        checkpoint_id = load_checkpoint(checkpoint_file)
        start_id = max(args.start_id, checkpoint_id + 1)
        if checkpoint_id > 0:
            logger.info(f"Resuming from checkpoint: last completed was id{checkpoint_id:05d}")
            logger.info(f"Starting from id{start_id:05d}")
        else:
            logger.info(f"No checkpoint found, starting from id{start_id:05d}")
    
    if start_id > args.end_id:
        logger.info("All instances already downloaded!")
        sys.exit(0)
    
    logger.info(f"Range: id{start_id:05d} to id{args.end_id:05d}")
    logger.info(f"Total instances to process: {args.end_id - start_id + 1}")
    
    # Main download loop
    completed = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    
    try:
        for instance_id in tqdm(range(start_id, args.end_id + 1), desc="Downloading"):
            instance_str = f"id{instance_id:05d}"
            
            # Check if already complete (double-check)
            files = get_instance_files(args.modality, instance_id, logger)
            instance_dir = args.data_root / args.modality / instance_str
            
            if is_instance_complete(instance_dir, files):
                logger.debug(f"{instance_str} already complete, skipping")
                skipped += 1
                save_checkpoint(checkpoint_file, instance_id)
                continue
            
            # Download this instance
            success = download_instance(
                args.modality,
                instance_id,
                args.data_root,
                args.connections,
                args.max_retries,
                logger,
            )
            
            if success:
                completed += 1
                save_checkpoint(checkpoint_file, instance_id)
                logger.info(f"✓ Completed {instance_str} ({completed} total)")
            else:
                failed += 1
                logger.error(f"✗ Failed {instance_str}")
                # Continue to next instance even if one fails
    
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        logger.info(f"Checkpoint saved: last completed was id{load_checkpoint(checkpoint_file):05d}")
        logger.info("Run again to resume from where you left off")
        sys.exit(1)
    
    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    logger.info(f"Completed: {completed} instances")
    logger.info(f"Skipped (already had): {skipped} instances")
    logger.info(f"Failed: {failed} instances")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Rate: {completed/(elapsed/3600):.1f} instances/hour")
    
    if failed > 0:
        logger.warning(f"\n{failed} instances failed. Run again to retry failed downloads.")
        sys.exit(1)
    else:
        logger.info("\nAll instances downloaded successfully!")
        # Clean up checkpoint file
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info(f"Removed checkpoint file: {checkpoint_file}")
        sys.exit(0)


if __name__ == "__main__":
    main()
