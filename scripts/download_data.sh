#!/bin/bash
# Download HEAT dataset from LANL Oceans11
# Supports parallel download with aria2c, wget, or curl
# HPC-ready with optional Slurm directives

set -euo pipefail

# ==================== CONFIGURATION ====================
# Uncomment below lines for Slurm HPC deployment
# #SBATCH --job-name=heat_download
# #SBATCH --output=download_%j.out
# #SBATCH --error=download_%j.err
# #SBATCH --time=04:00:00
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1

# Default URLs from plan.md
CYL_URL="https://oceans11.lanl.gov/heat/cyl/cx241203_fp16_full.tar"
PLI_URL="https://oceans11.lanl.gov/heat/pli/pli240420.tar"

# Default output directories
DATA_ROOT="${DATA_ROOT:-./data}"
CYL_DIR="${DATA_ROOT}/cyl"
PLI_DIR="${DATA_ROOT}/pli"

# Download tool preference: aria2c > wget > curl
DOWNLOAD_TOOL="${DOWNLOAD_TOOL:-auto}"

# Parallel connections for aria2c
ARIA2C_CONNECTIONS="${ARIA2C_CONNECTIONS:-4}"

# ==================== LOGGING ====================
log_info() { echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }
log_error() { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2; }

# ==================== DOWNLOAD FUNCTIONS ====================

detect_download_tool() {
    if command -v aria2c &> /dev/null; then
        echo "aria2c"
    elif command -v wget &> /dev/null; then
        echo "wget"
    elif command -v curl &> /dev/null; then
        echo "curl"
    else
        log_error "No download tool found. Please install aria2c, wget, or curl."
        exit 1
    fi
}

download_file() {
    local url="$1"
    local output_path="$2"
    local tool="${3:-$(detect_download_tool)}"

    mkdir -p "$(dirname "$output_path")"

    log_info "Downloading: $url"
    log_info "Tool: $tool, Output: $output_path"

    case "$tool" in
        aria2c)
            aria2c \
                --max-connection-per-server="$ARIA2C_CONNECTIONS" \
                --split="$ARIA2C_CONNECTIONS" \
                --min-split-size=1M \
                --continue=true \
                --file-allocation=none \
                --out="$(basename "$output_path")" \
                --dir="$(dirname "$output_path")" \
                "$url"
            ;;
        wget)
            wget \
                --continue \
                --tries=3 \
                --timeout=60 \
                --progress=bar:force \
                -O "$output_path" \
                "$url"
            ;;
        curl)
            curl \
                --continue-at - \
                --retry 3 \
                --retry-delay 5 \
                --max-time 3600 \
                -L -o "$output_path" \
                "$url"
            ;;
        *)
            log_error "Unknown download tool: $tool"
            exit 1
            ;;
    esac
}

extract_tar() {
    local tar_path="$1"
    local extract_dir="$2"

    log_info "Extracting: $tar_path to $extract_dir"
    mkdir -p "$extract_dir"

    # Use pigz for parallel decompression if available
    if command -v pigz &> /dev/null && command -v pv &> /dev/null; then
        pv "$tar_path" | pigz -dc | tar -xf - -C "$extract_dir"
    else
        tar -xzf "$tar_path" -C "$extract_dir"
    fi
}

# ==================== MAIN ====================

main() {
    log_info "HEAT Dataset Download Starting"
    log_info "Data root: $DATA_ROOT"

    # Detect download tool
    if [[ "$DOWNLOAD_TOOL" == "auto" ]]; then
        DOWNLOAD_TOOL=$(detect_download_tool)
    fi
    log_info "Using download tool: $DOWNLOAD_TOOL"

    # Create directories
    mkdir -p "$CYL_DIR" "$PLI_DIR"

    # Download CYL data
    cyl_tar="$DATA_ROOT/cx241203_fp16_full.tar.gz"
    if [[ ! -f "$cyl_tar" ]]; then
        cyl_tar="${cyl_tar%.gz}"  # Try without .gz extension
    fi

    if [[ ! -f "$cyl_tar" && ! -f "$cyl_tar.gz" ]]; then
        download_file "$CYL_URL" "$cyl_tar" "$DOWNLOAD_TOOL"
    else
        log_info "CYL tar already exists, skipping download"
    fi

    # Download PLI data
    pli_tar="$DATA_ROOT/pli240420.tar.gz"
    if [[ ! -f "$pli_tar" ]]; then
        pli_tar="${pli_tar%.gz}"
    fi

    if [[ ! -f "$pli_tar" && ! -f "$pli_tar.gz" ]]; then
        download_file "$PLI_URL" "$pli_tar" "$DOWNLOAD_TOOL"
    else
        log_info "PLI tar already exists, skipping download"
    fi

    # Extract CYL data
    if [[ -f "$cyl_tar" || -f "$cyl_tar.gz" ]]; then
        if [[ ! -d "$CYL_DIR/id00001" ]]; then
            extract_tar "$cyl_tar" "$CYL_DIR"
            log_info "CYL extraction complete"
        else
            log_info "CYL data already extracted, skipping"
        fi
    fi

    # Extract PLI data
    if [[ -f "$pli_tar" || -f "$pli_tar.gz" ]]; then
        if [[ ! -d "$PLI_DIR/id00001" ]]; then
            extract_tar "$pli_tar" "$PLI_DIR"
            log_info "PLI extraction complete"
        else
            log_info "PLI data already extracted, skipping"
        fi
    fi

    log_info "Download and extraction complete!"
    log_info "CYL data: $CYL_DIR"
    log_info "PLI data: $PLI_DIR"
}

# ==================== CLI INTERFACE ====================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Download HEAT dataset from LANL Oceans11 for surrogate model training.

OPTIONS:
    -h, --help              Show this help message
    -m, --modality {cyl|pli|both}  Download specific modality (default: both)
    -o, --output DIR        Output directory (default: ./data)
    -t, --tool {auto|aria2c|wget|curl}  Download tool (default: auto)
    -c, --connections N      Parallel connections for aria2c (default: 4)
    --no-extract            Download only, skip extraction

ENVIRONMENT VARIABLES:
    DATA_ROOT               Output directory (overridden by -o)
    DOWNLOAD_TOOL           Download tool (overridden by -t)
    ARIA2C_CONNECTIONS      Parallel connections (overridden by -c)

EXAMPLES:
    # Download both modalities with auto-detected tool
    $0

    # Download only CYL with aria2c (8 connections)
    $0 -m cyl -t aria2c -c 8

    # Download to custom location
    $0 -o /scratch/user/heat_data

HPC USAGE:
    # Add Slurm directives at top of this script, then:
    sbatch $0

EOF
}

# Parse arguments
MODALITY="both"
NO_EXTRACT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -m|--modality)
            MODALITY="$2"
            shift 2
            ;;
        -o|--output)
            DATA_ROOT="$2"
            shift 2
            ;;
        -t|--tool)
            DOWNLOAD_TOOL="$2"
            shift 2
            ;;
        -c|--connections)
            ARIA2C_CONNECTIONS="$2"
            shift 2
            ;;
        --no-extract)
            NO_EXTRACT=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate modality
if [[ "$MODALITY" != "cyl" && "$MODALITY" != "pli" && "$MODALITY" != "both" ]]; then
    log_error "Invalid modality: $MODALITY (must be cyl, pli, or both)"
    exit 1
fi

# Run main
main
