#!/usr/bin/env python3
"""
Selective instance downloader for HEAT dataset using aria2c.

Downloads individual simulation instances (id00001-id02100) rather than full tar,
saving bandwidth and storage. Uses aria2c for parallel downloads with resume support.

URL pattern for instances:
- CYL: https://oceans11.lanl.gov/heat/cyl/cx241203_fp16_full/id{ID}/
- PLI: https://oceans11.lanl.gov/heat/pli/pli240420/id{ID}/

Files per instance:
- CYL: cx241203_id{ID}_pvi_idx{IDX}.npz (idx00000 to ~idx00060)
- PLI: pli240420_id{ID}_pvi_idx{IDX}.npz (idx00000 to ~idx00100)
"""

import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple
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

# Default instance range
DEFAULT_START_ID = 1
DEFAULT_END_ID = 2100


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def fetch_directory_listing(url: str, logger: logging.Logger) -> List[str]:
    """
    Fetch Apache directory listing and extract file links.
    
    Returns:
        List of filenames found in directory
    """
    logger.debug(f"Fetching directory: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return []
    
    # Parse HTML for file links
    html = response.text
    
    # Find all href links
    pattern = r'href="([^"]+)"'
    links = re.findall(pattern, html)
    
    # Filter to actual files (not directories, parent, or sort links)
    files = []
    for link in links:
        # Skip parent directory, sort links, and directories
        if link in ("../", "?C=N;O=D", "?C=M;O=A", "?C=S;O=A", "?C=D;O=A"):
            continue
        if link.endswith("/"):
            continue  # Skip subdirectories
        files.append(link)
    
    logger.debug(f"Found {len(files)} files in {url}")
    return files


def get_instance_files(modality: str, instance_id: int, logger: logging.Logger) -> List[str]:
    """
    Get list of NPZ files for a specific instance.
    
    Args:
        modality: 'cyl' or 'pli'
        instance_id: Instance number (1-2100)
        logger: Logger instance
    
    Returns:
        List of filenames
    """
    base_url = BASE_URLS[modality]
    instance_str = f"id{instance_id:05d}"
    instance_url = urljoin(base_url, f"{instance_str}/")
    
    files = fetch_directory_listing(instance_url, logger)
    
    # Filter to NPZ files matching pattern
    pattern = re.compile(FILE_PATTERNS[modality])
    npz_files = [f for f in files if pattern.match(f)]
    
    logger.debug(f"Instance {instance_str}: {len(npz_files)} NPZ files")
    return npz_files


def generate_url_list(
    modality: str,
    start_id: int,
    end_id: int,
    data_root: Path,
    logger: logging.Logger,
) -> Tuple[Path, int, int]:
    """
    Generate aria2c URL list file for specified instances.
    
    Checks for existing files to support resume.
    
    Returns:
        (url_list_path, total_files, existing_files)
    """
    base_url = BASE_URLS[modality]
    modality_dir = data_root / modality
    
    # Create output directory
    modality_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate URL list
    url_list_file = data_root / f"{modality}_urls_{start_id:05d}_{end_id:05d}.txt"
    
    total_files = 0
    existing_files = 0
    
    logger.info(f"Generating URL list for {modality} instances {start_id}-{end_id}")
    
    with open(url_list_file, "w") as f:
        for instance_id in tqdm(range(start_id, end_id + 1), desc=f"Scanning {modality}"):
            instance_str = f"id{instance_id:05d}"
            instance_dir = modality_dir / instance_str
            
            # Get files for this instance
            files = get_instance_files(modality, instance_id, logger)
            
            for filename in files:
                total_files += 1
                
                # Check if file already exists
                local_path = instance_dir / filename
                if local_path.exists():
                    existing_files += 1
                    logger.debug(f"Skipping existing: {local_path}")
                    continue
                
                # Write URL to list
                file_url = urljoin(base_url, f"{instance_str}/{filename}")
                # aria2c format: URL with output path
                out_path = str(instance_dir / filename)
                f.write(f"{file_url}\n  out={out_path}\n")
    
    logger.info(f"URL list: {url_list_file}")
    logger.info(f"Total files: {total_files}, Existing: {existing_files}, To download: {total_files - existing_files}")
    
    return url_list_file, total_files, existing_files


def run_aria2c(
    url_list: Path,
    output_dir: Path,
    connections: int,
    max_concurrent: int,
    logger: logging.Logger,
) -> bool:
    """
    Run aria2c with URL list.
    
    Args:
        url_list: Path to aria2c input file
        output_dir: Base output directory
        connections: Connections per server
        max_concurrent: Max concurrent downloads
        logger: Logger
    
    Returns:
        True if successful
    """
    # Check aria2c is available
    try:
        result = subprocess.run(["aria2c", "--version"], capture_output=True, text=True)
        logger.info(f"Using: {result.stdout.split(chr(10))[0]}")
    except FileNotFoundError:
        logger.error("aria2c not found. Please install aria2c.")
        return False
    
    # Build aria2c command
    cmd = [
        "aria2c",
        "--input-file", str(url_list),
        "--dir", str(output_dir),
        "--max-connection-per-server", str(connections),
        "--split", str(connections),
        "--max-concurrent-downloads", str(max_concurrent),
        "--continue", "true",  # Resume support
        "--remote-time", "true",
        "--auto-file-renaming", "false",
        "--allow-overwrite", "false",
        "--conditional-get", "true",  # Don't re-download if file exists and matches
        "--log-level", "warn",
        "--summary-interval", "30",
        "--console-log-level", "warn",
    ]
    
    logger.info(f"Starting aria2c: {' '.join(cmd)}")
    logger.info(f"Connections per server: {connections}, Max concurrent: {max_concurrent}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        logger.info(f"Download complete in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"aria2c failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user. Run again to resume.")
        return False


def verify_download(
    modality: str,
    start_id: int,
    end_id: int,
    data_root: Path,
    logger: logging.Logger,
) -> Tuple[int, int, int]:
    """
    Verify downloaded files.
    
    Returns:
        (expected_instances, complete_instances, total_files)
    """
    modality_dir = data_root / modality
    
    expected_instances = end_id - start_id + 1
    complete_instances = 0
    total_files = 0
    
    logger.info(f"Verifying {modality} downloads...")
    
    for instance_id in tqdm(range(start_id, end_id + 1), desc="Verifying"):
        instance_str = f"id{instance_id:05d}"
        instance_dir = modality_dir / instance_str
        
        if not instance_dir.exists():
            logger.warning(f"Missing instance directory: {instance_dir}")
            continue
        
        npz_files = list(instance_dir.glob("*.npz"))
        
        if len(npz_files) > 0:
            complete_instances += 1
            total_files += len(npz_files)
    
    logger.info(f"Verification complete:")
    logger.info(f"  Expected instances: {expected_instances}")
    logger.info(f"  Complete instances: {complete_instances}")
    logger.info(f"  Total NPZ files: {total_files}")
    
    return expected_instances, complete_instances, total_files


def main():
    parser = argparse.ArgumentParser(
        description="Selective HEAT instance downloader using aria2c",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all 2100 CYL instances
    python scripts/download_selective.py --modality cyl --data-root ./data

    # Download specific range (for parallel jobs)
    python scripts/download_selective.py --modality pli --start-id 1 --end-id 500 --data-root ./data_part1

    # Resume interrupted download (automatic)
    python scripts/download_selective.py --modality cyl --data-root ./data

    # Dry run - generate URL list only
    python scripts/download_selective.py --modality cyl --dry-run --data-root ./data

    # Fast mode (more parallel connections)
    python scripts/download_selective.py --modality cyl --connections 8 --max-concurrent 32

Note:
    - This script downloads ~150GB for 2100 instances vs 2.6TB for full tar
    - aria2c supports resume - run again to continue interrupted downloads
    - Existing files are checked and skipped automatically
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
        default=DEFAULT_START_ID,
        help=f"Starting instance ID (default: {DEFAULT_START_ID})",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=DEFAULT_END_ID,
        help=f"Ending instance ID (default: {DEFAULT_END_ID})",
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=4,
        help="Connections per server for aria2c (default: 4)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="Maximum concurrent downloads for aria2c (default: 16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate URL list only, don't download",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloads after completion",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification step",
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
    logger.info("HEAT Selective Instance Downloader (aria2c)")
    logger.info("=" * 60)
    logger.info(f"Modality: {args.modality}")
    logger.info(f"Instance range: {args.start_id:05d} - {args.end_id:05d}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Expected instances: {args.end_id - args.start_id + 1}")
    
    # Validate range
    if args.start_id < 1 or args.end_id > 99999:
        logger.error("Instance IDs must be between 1 and 99999")
        sys.exit(1)
    if args.start_id > args.end_id:
        logger.error("Start ID must be <= end ID")
        sys.exit(1)
    
    # Generate URL list
    url_list, total_files, existing_files = generate_url_list(
        args.modality,
        args.start_id,
        args.end_id,
        args.data_root,
        logger,
    )
    
    # Check if all files already exist
    if total_files > 0 and existing_files == total_files:
        logger.info("All files already exist! Nothing to download.")
        if args.verify:
            verify_download(
                args.modality,
                args.start_id,
                args.end_id,
                args.data_root,
                logger,
            )
        sys.exit(0)
    
    # Dry run - just generate URL list
    if args.dry_run:
        logger.info(f"Dry run complete. URL list: {url_list}")
        logger.info("Use aria2c manually:")
        logger.info(f"  aria2c --input-file {url_list} -j {args.max_concurrent}")
        sys.exit(0)
    
    # Run aria2c
    success = run_aria2c(
        url_list,
        args.data_root,
        args.connections,
        args.max_concurrent,
        logger,
    )
    
    if not success:
        logger.error("Download incomplete. Run again to resume.")
        sys.exit(1)
    
    # Verification
    if not args.skip_verify:
        expected, complete, files = verify_download(
            args.modality,
            args.start_id,
            args.end_id,
            args.data_root,
            logger,
        )
        
        if complete < expected:
            logger.warning(f"Only {complete}/{expected} instances complete")
            missing = expected - complete
            logger.info(f"Run again to download {missing} missing instances")
        else:
            logger.info(f"All {expected} instances verified successfully!")
    
    logger.info("=" * 60)
    logger.info("Download complete!")
    logger.info(f"Data location: {args.data_root / args.modality}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
