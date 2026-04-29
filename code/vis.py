#!/usr/bin/env python3
"""HEAT .npz visualization per specs/plan.md Visualization section.

CYL and PLI share one script. Each figure uses a *single* target scalar field on
the R–Z grid; that field sets the color gradient everywhere. VoF masks define
material regions (legend + borders). PLI defaults to inverted vertical axis.
When rendering multiple snapshots, the colorbar range is fixed for the whole
animation. Known targets (av_density, av_temperature, av_pressure, Uvelocity,
Wvelocity) default to physics-based limits and unit labels; other targets use
the first frame (2–98th percentiles, vmin clamped to 0). Use --scale percentile
to force data-driven limits for any target.

Outputs under <project>/out/<dataset>/<run_id>_vis/
  - one PNG per timestep: <stem>.gif from --target gradient
  - optional <target>.gif

By default the R–Z panel is **mirrored across R=0** (reflected field on the left,
original half-slice on the right) so the whole figure is symmetric. Use --no-mirror
for a single half-slice only. The colorbar and material legend sit in their own
columns so they do not cover the field.

Pass ``-t all`` for one row of three panels (velocity = |v| with quiver, density,
pressure); outputs under ``..._vis/all/`` with ``all.gif``. Use ``-t velocity`` for
the combined velocity view alone (GIF ``velocity.gif``). Temperature stays available
for ``-t av_temperature`` but is omitted from ``all``.

No config files — only command-line parameters.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image


# ── VoF masks used to segment the grid (plan.md) ──────────────────────────

COMPONENT_MASKS_CYL: list[tuple[str, str]] = [
    ("Main charge (void)", "vofm_Void"),
    ("Main charge", "vofm_maincharge"),
    ("Booster", "vofm_booster"),
    ("Wall", "vofm_wall"),
]

COMPONENT_MASKS_PLI: list[tuple[str, str]] = [
    ("Main charge (void)", "vofm_Void"),
    ("Main charge", "vofm_maincharge"),
    ("Booster — throw", "vofm_throw"),
    ("Booster — striker", "vofm_striker"),
    ("Booster — cushion", "vofm_cushion"),
    ("Wall", "vofm_case"),
    ("Outside air", "vofm_outside_air"),
]

TWO_COLOR_GRADIENT = "plasma"
THREE_COLOR_GRADIENT = "seismic"

# Inferred HEAT field units (no unit metadata in this repo’s specs; values follow
# typical LANL hydrodynamics exports and the magnitudes you described: density O(1)
# g/cm³, temperature O(10³) K, pressure O(0.1–10) GPa). Verify against the HEAT
# dataset README when available.
# Tuple: (unit string for colorbar, vmax for physics scale, symmetric_velocity).
# Velocity components use vmin = −vmax, vmax = +vmax (signed flow).
TARGET_FIELD_PHYSICS: dict[str, tuple[str, float, bool]] = {
    "av_density": ("g/cm³", 12.0, False),
    "av_temperature": ("K", 2000.0, False),
    "av_pressure": ("GPa", 0.3, False),
    "Uvelocity": ("m/s", 1.0, True),
    "Wvelocity": ("m/s", 1.0, True),
    # Speed magnitude √(U²+W²); vmax ≈ √2 × component cap when both axes saturate.
    "velocity": ("m/s", 1.4142135623730951, False),
}

# ``-t all``: one row — velocity (|v|, quiver), density, pressure (no temperature).
ALL_PANEL_TARGETS: list[str] = [
    "velocity",
    "av_density",
    "av_pressure",
]

# Subsample grid for quiver (~N/QUIVER_SUBSAMPLE_TARGET steps along each axis).
QUIVER_SUBSAMPLE_TARGET = 22

ALL_PANEL_SUBDIR = "all"


# ── paths ─────────────────────────────────────────────────────────────────

def select_npz_files(files: list[str], max_frames: int | None, policy: str) -> list[str]:
    if not files:
        return files
    if max_frames is None or max_frames <= 0 or len(files) <= max_frames:
        return files
    n = max_frames
    if policy == "first":
        return files[:n]
    idx = np.unique(np.round(np.linspace(0, len(files) - 1, n)).astype(int))
    return [files[i] for i in idx]


def find_project_root(start_path: str) -> str:
    cur = os.path.abspath(start_path)
    if not os.path.isdir(cur):
        cur = os.path.dirname(cur)
    for _ in range(40):
        if os.path.isdir(os.path.join(cur, "data")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.getcwd()


def resolve_output_under_out(
    input_dir: str,
    dataset: str,
    override: str | None,
) -> str:
    root = find_project_root(input_dir)
    run_id = os.path.basename(os.path.abspath(input_dir).rstrip(os.sep))
    out_root = os.path.abspath(os.path.join(root, "out"))
    if override:
        path = override if os.path.isabs(override) else os.path.normpath(
            os.path.join(root, override)
        )
    else:
        path = os.path.join(out_root, dataset, f"{run_id}_vis")
    path = os.path.abspath(os.path.normpath(path))
    if path != out_root and not path.startswith(out_root + os.sep):
        sys.exit(
            f"Output must be under {out_root} (refusing {path}). "
            "Use a path under out/ or omit -o for the default layout."
        )
    return path


def assert_under_out(path: str, purpose: str = "paths") -> None:
    root = find_project_root(path)
    out_root = os.path.abspath(os.path.join(root, "out"))
    ap = os.path.abspath(path)
    if ap != out_root and not ap.startswith(out_root + os.sep):
        sys.exit(f"{purpose} must be under {out_root}, got {ap}")


def detect_dataset_kind(input_dir: str) -> str | None:
    parts = os.path.abspath(input_dir).split(os.sep)
    if "cyl" in parts:
        i = parts.index("cyl")
        if i > 0 and parts[i - 1] == "data":
            return "cyl"
    if "pli" in parts:
        i = parts.index("pli")
        if i > 0 and parts[i - 1] == "data":
            return "pli"
    return None


# ── grids & segmentation ──────────────────────────────────────────────────


def build_component_labels(
    data: np.lib.npyio.NpzFile,
    masks: list[tuple[str, str]],
) -> tuple[np.ndarray, list[str]]:
    mats: list[np.ndarray] = []
    names: list[str] = []
    for disp, key in masks:
        if key not in data:
            sys.exit(f"Missing VoF mask key {key!r} for this snapshot.")
        m = np.nan_to_num(data[key].astype(np.float64))
        mats.append(m)
        names.append(disp)
    if not mats:
        sys.exit("No VoF masks configured.")
    stack = np.stack(mats, axis=-1)
    dominant = np.argmax(stack, axis=-1)
    inactive = stack.max(axis=-1) < 1e-9
    dominant = dominant.astype(np.float64)
    dominant[inactive] = np.nan
    return dominant, names


def edges_from_labels(labels: np.ndarray) -> np.ndarray:
    L = np.where(np.isfinite(labels), labels, -1).astype(np.int32)
    edge = np.zeros_like(L, dtype=bool)
    edge[1:, :] |= L[1:, :] != L[:-1, :]
    edge[:-1, :] |= L[1:, :] != L[:-1, :]
    edge[:, 1:] |= L[:, 1:] != L[:, :-1]
    edge[:, :-1] |= L[:, 1:] != L[:, :-1]
    return edge


def mirror_r_half_slice(
    rcoord: np.ndarray,
    fld: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reflect the half-domain across ``R = 0`` so *R* spans negative and positive values.

    The original slice (typically ``R >= 0``) stays on the **right**; the **left** half
    is the horizontal mirror (same field values) for a symmetric full cross-section.
    If ``Rcoord[0]`` is (nearly) zero, that column is not duplicated at the join.
    """
    rcoord = np.asarray(rcoord, dtype=np.float64)
    nr = rcoord.size
    nz, nc = fld.shape[0], fld.shape[1]
    if nr < 2 or nc != nr or labels.shape != (nz, nr):
        return rcoord, fld, labels

    fl_f = np.fliplr(fld)
    fl_l = np.fliplr(labels)
    r0 = float(rcoord[0])
    span = max(
        float(np.nanmax(np.abs(rcoord)) - np.nanmin(np.abs(rcoord))),
        float(abs(rcoord[-1] - r0)),
        1.0,
    )
    atol = max(1e-12, 1e-7 * span)
    if abs(r0) <= atol:
        r_left = -rcoord[::-1][:-1]
        fld_m = np.hstack((fl_f[:, :-1], fld))
        lab_m = np.hstack((fl_l[:, :-1], labels))
        r_out = np.concatenate((r_left, rcoord))
    else:
        r_left = -rcoord[::-1]
        fld_m = np.hstack((fl_f, fld))
        lab_m = np.hstack((fl_l, labels))
        r_out = np.concatenate((r_left, rcoord))
    return r_out, fld_m, lab_m


def mirror_vector_r_half_slice(
    rcoord: np.ndarray,
    u: np.ndarray,
    w: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror half-slice across ``R = 0``: ``U`` (R-component) flips sign on ``R < 0``; ``W`` mirrors like a scalar."""
    rcoord = np.asarray(rcoord, dtype=np.float64)
    nr = rcoord.size
    nz, nc = u.shape[0], u.shape[1]
    if nr < 2 or nc != nr or w.shape != u.shape or labels.shape != u.shape:
        return rcoord, u, w, labels

    fl_u = np.fliplr(u)
    fl_w = np.fliplr(w)
    fl_l = np.fliplr(labels)
    r0 = float(rcoord[0])
    span = max(
        float(np.nanmax(np.abs(rcoord)) - np.nanmin(np.abs(rcoord))),
        float(abs(rcoord[-1] - r0)),
        1.0,
    )
    atol = max(1e-12, 1e-7 * span)
    if abs(r0) <= atol:
        r_left = -rcoord[::-1][:-1]
        u_m = np.hstack((-fl_u[:, :-1], u))
        w_m = np.hstack((fl_w[:, :-1], w))
        lab_m = np.hstack((fl_l[:, :-1], labels))
        r_out = np.concatenate((r_left, rcoord))
    else:
        r_left = -rcoord[::-1]
        u_m = np.hstack((-fl_u, u))
        w_m = np.hstack((fl_w, w))
        lab_m = np.hstack((fl_l, labels))
        r_out = np.concatenate((r_left, rcoord))
    return r_out, u_m, w_m, lab_m


def _align_z_vertical(
    zcoord: np.ndarray,
    fld: np.ndarray,
    labels: np.ndarray,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put *Z* on the axis so values increase upward (e.g. −4 … 24, not 24 … −4).

    1. If *Zcoord* decreases along the field row index, flip rows so it increases.
    2. PLI snapshots are stored with rows top-to-bottom vs the desired R–Z view;
       reflect once so the vertical axis reads from smaller *Z* at the bottom to
       larger *Z* at the top (matches typical ``0 → 20`` reading order).
    """
    z = np.asarray(zcoord, dtype=np.float64)
    f, lab = fld, labels
    if len(z) >= 2 and float(z[0]) > float(z[-1]):
        return z[::-1].copy(), np.flipud(f), np.flipud(lab)
    if dataset == "pli" and len(z) >= 2:
        return z[::-1].copy(), np.flipud(f), np.flipud(lab)
    return z, f, lab


def aligned_target_field(
    data: np.lib.npyio.NpzFile,
    dataset: str,
    masks: list[tuple[str, str]],
    target: str,
) -> np.ndarray:
    """Target scalar on the Z–R grid after the same row flips as ``plot_snapshot``."""
    if target not in data:
        sys.exit(f"Missing target array {target!r}.")
    fld = np.nan_to_num(data[target].astype(np.float64))
    zcoord = np.asarray(data["Zcoord"], dtype=np.float64)
    rcoord = np.asarray(data["Rcoord"], dtype=np.float64)
    if fld.shape != (len(zcoord), len(rcoord)):
        sys.exit(
            f"Shape mismatch: {target} is {fld.shape}, "
            f"expected ({len(zcoord)}, {len(rcoord)}) from Zcoord × Rcoord."
        )
    labels, _ = build_component_labels(data, masks)
    _z, fld, _lab = _align_z_vertical(zcoord, fld, labels, dataset)
    return fld


def aligned_u_w_velocity(
    data: np.lib.npyio.NpzFile,
    dataset: str,
    masks: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aligned ``Zcoord``, ``Rcoord``, ``Uvelocity``, ``Wvelocity``, VoF labels (same row order)."""
    if "Uvelocity" not in data or "Wvelocity" not in data:
        sys.exit("Need both Uvelocity and Wvelocity in the snapshot.")
    labels_u, _ = build_component_labels(data, masks)
    zcoord = np.asarray(data["Zcoord"], dtype=np.float64)
    rcoord = np.asarray(data["Rcoord"], dtype=np.float64)
    u = np.nan_to_num(data["Uvelocity"].astype(np.float64))
    w = np.nan_to_num(data["Wvelocity"].astype(np.float64))
    if u.shape != (len(zcoord), len(rcoord)) or w.shape != u.shape:
        sys.exit(
            f"Uvelocity/Wvelocity shape {u.shape} vs ({len(zcoord)}, {len(rcoord)})."
        )
    zcoord, u, labels_u = _align_z_vertical(zcoord, u, labels_u, dataset)
    labels_w, _ = build_component_labels(data, masks)
    z_w = np.asarray(data["Zcoord"], dtype=np.float64)
    _, w, _ = _align_z_vertical(z_w, w, labels_w, dataset)
    return zcoord, rcoord, u, w, labels_u


def compute_intensity_limits(fld: np.ndarray) -> tuple[float, float]:
    """2–98th percentile range (same rule as before); used to lock the color scale."""
    finite = fld[np.isfinite(fld)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [2.0, 98.0])
    if vmax <= vmin:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    return float(vmin), float(vmax)


def colormap_and_norm(
    vmin: float, vmax: float
) -> tuple[str, Normalize | TwoSlopeNorm]:
    """Return colormap name and norm (diverging if ``vmin < 0``)."""
    if vmin < 0:
        cmap = THREE_COLOR_GRADIENT
        if vmax > 0:
            norm: Normalize | TwoSlopeNorm = TwoSlopeNorm(
                vmin=vmin, vcenter=0.0, vmax=vmax
            )
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = TWO_COLOR_GRADIENT
        norm = Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def resolve_animation_color_limits(
    target: str,
    dataset: str,
    masks: list[tuple[str, str]],
    npz_paths: list[str],
    scale_mode: str,
) -> tuple[float, float, str]:
    """Return fixed vmin, vmax, and colorbar/title label for the whole run.

    ``physics`` — use TARGET_FIELD_PHYSICS when the target is listed; else first-frame
    percentiles. vmin ≥ 0 for non-velocity unknown fields.
    ``percentile`` — first frame 2–98% for all targets; vmin ≥ 0 except for
    Uvelocity / Wvelocity (signed).
    """
    if scale_mode == "physics" and target == "velocity":
        unit, vmax_phys, _ = TARGET_FIELD_PHYSICS["velocity"]
        label = f"|v| = √(U²+W²) ({unit})"
        return 0.0, vmax_phys, label

    if scale_mode == "physics" and target in TARGET_FIELD_PHYSICS:
        unit, vmax_phys, symmetric = TARGET_FIELD_PHYSICS[target]
        label = f"{target} ({unit})"
        if symmetric:
            return -vmax_phys, vmax_phys, label
        return 0.0, vmax_phys, label

    if target == "velocity":
        first_data = np.load(npz_paths[0])
        try:
            _z, _r, u, w, _lab = aligned_u_w_velocity(first_data, dataset, masks)
            speed = np.sqrt(u * u + w * w)
            lim_vmin, lim_vmax = compute_intensity_limits(speed)
        finally:
            first_data.close()
        lim_vmin = max(0.0, lim_vmin)
        return lim_vmin, lim_vmax, "|v| = √(U²+W²) (m/s)"

    first_data = np.load(npz_paths[0])
    try:
        fld0 = aligned_target_field(first_data, dataset, masks, target)
        lim_vmin, lim_vmax = compute_intensity_limits(fld0)
    finally:
        first_data.close()
    if target not in ("Uvelocity", "Wvelocity"):
        lim_vmin = max(0.0, lim_vmin)
    return lim_vmin, lim_vmax, target


def plot_snapshot(
    data: np.lib.npyio.NpzFile,
    dataset: str,
    masks: list[tuple[str, str]],
    target: str,
    invert_z: bool,
    stem: str,
    out_png: str,
    border_lw_pts: float = 2.5,
    vmin: float | None = None,
    vmax: float | None = None,
    field_label: str | None = None,
    mirror_r: bool = True,
) -> None:
    if target not in data:
        sys.exit(f"Missing target array {target!r}.")

    fld = np.nan_to_num(data[target].astype(np.float64))
    rcoord = np.asarray(data["Rcoord"], dtype=np.float64)
    zcoord = np.asarray(data["Zcoord"], dtype=np.float64)
    if fld.shape != (len(zcoord), len(rcoord)):
        sys.exit(
            f"Shape mismatch: {target} is {fld.shape}, "
            f"expected ({len(zcoord)}, {len(rcoord)}) from Zcoord × Rcoord."
        )

    labels, mask_names = build_component_labels(data, masks)
    zcoord, fld, labels = _align_z_vertical(zcoord, fld, labels, dataset)
    if mirror_r:
        rcoord, fld, labels = mirror_r_half_slice(rcoord, fld, labels)

    edges = edges_from_labels(labels)

    RG, ZG = np.meshgrid(rcoord, zcoord)

    n_reg = len(mask_names)
    region_colors = np.array(
        [plt.cm.tab10(i / max(n_reg - 1, 1) * 0.95) for i in range(n_reg)],
        dtype=np.float64,
    )

    if vmin is None and vmax is None:
        vmin, vmax = compute_intensity_limits(fld)
    elif vmin is None or vmax is None:
        sys.exit("plot_snapshot: pass both vmin and vmax, or neither.")

    cmap, norm = colormap_and_norm(vmin, vmax)

    fig, (ax, cax, leg_ax) = plt.subplots(
        ncols=3,
        figsize=(12.0, 8.0),
        gridspec_kw={"width_ratios": [1.0, 0.05, 0.36], "wspace": 0.45},
    )
    leg_ax.set_axis_off()

    pcm = ax.pcolormesh(
        RG,
        ZG,
        fld,
        shading="auto",
        cmap=cmap,
        norm=norm,
        antialiased=False,
    )

    zi, ri = np.where(edges)
    ec = np.zeros((len(zi), 4), dtype=np.float64)
    for k in range(len(zi)):
        lb = labels[zi[k], ri[k]]
        if np.isfinite(lb):
            idx = int(np.clip(lb, 0, n_reg - 1))
            ec[k] = region_colors[idx]
        else:
            ec[k] = (0.25, 0.25, 0.25, 1.0)
    ax.scatter(
        rcoord[ri],
        zcoord[zi],
        c=ec,
        s=(border_lw_pts * 2.2) ** 2 / 16.0,
        marker="s",
        linewidths=0,
        zorder=10,
        rasterized=True,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_xlim(float(np.min(rcoord)), float(np.max(rcoord)))

    z_min, z_max = float(np.min(zcoord)), float(np.max(zcoord))
    ax.set_ylim(z_min, z_max)
    if invert_z:
        ax.invert_yaxis()

    display = field_label if field_label is not None else target
    ttl = f"{stem} — {dataset.upper()} — {display}"
    ax.set_title(ttl, fontsize=11)

    cb = fig.colorbar(pcm, cax=cax, orientation="vertical")
    cb.set_label(display)

    leg_handles = [
        mpatches.Patch(
            color=tuple(region_colors[i, :3]),
            label=name,
            alpha=float(region_colors[i, 3]),
        )
        for i, name in enumerate(mask_names)
    ]
    leg_ax.legend(
        handles=leg_handles,
        loc="center left",
        fontsize=8,
        framealpha=0.9,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.08)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    # print(f"  saved {out_png}")


def _add_subsampled_quiver(
    ax,
    rg: np.ndarray,
    zg: np.ndarray,
    u: np.ndarray,
    w: np.ndarray,
) -> None:
    """Direction-only arrows (unit vectors); magnitude is shown in the colormap."""
    nz, nr = u.shape
    sr = max(1, nr // QUIVER_SUBSAMPLE_TARGET)
    sz = max(1, nz // QUIVER_SUBSAMPLE_TARGET)
    uq = u[::sz, ::sr]
    wq = w[::sz, ::sr]
    rq = rg[::sz, ::sr]
    zq = zg[::sz, ::sr]
    sp = np.hypot(uq, wq)
    un = np.divide(uq, sp, out=np.zeros_like(uq), where=sp > 1e-30)
    wn = np.divide(wq, sp, out=np.zeros_like(wq), where=sp > 1e-30)
    ax.quiver(
        rq,
        zq,
        un,
        wn,
        angles="uv",
        scale_units="xy",
        scale=26.0,
        width=0.0035,
        color="0.12",
        alpha=0.72,
        zorder=5,
    )


def plot_snapshot_velocity(
    data: np.lib.npyio.NpzFile,
    dataset: str,
    masks: list[tuple[str, str]],
    invert_z: bool,
    stem: str,
    out_png: str,
    vmin: float,
    vmax: float,
    field_label: str | None,
    mirror_r: bool = True,
    border_lw_pts: float = 2.5,
) -> None:
    """|v| = √(U²+W²) as color; subsampled quiver shows (U, W) direction in the R–Z plane."""
    zcoord, rcoord, u, w, labels = aligned_u_w_velocity(data, dataset, masks)
    if mirror_r:
        rcoord, u, w, labels = mirror_vector_r_half_slice(rcoord, u, w, labels)
    speed = np.sqrt(u * u + w * w)
    RG, ZG = np.meshgrid(rcoord, zcoord)
    edges = edges_from_labels(labels)

    _, mask_names = build_component_labels(data, masks)
    n_reg = len(mask_names)
    region_colors = np.array(
        [plt.cm.tab10(i / max(n_reg - 1, 1) * 0.95) for i in range(n_reg)],
        dtype=np.float64,
    )

    cmap, norm = colormap_and_norm(vmin, vmax)

    fig, (ax, cax, leg_ax) = plt.subplots(
        ncols=3,
        figsize=(12.0, 8.0),
        gridspec_kw={"width_ratios": [1.0, 0.05, 0.36], "wspace": 0.45},
    )
    leg_ax.set_axis_off()

    pcm = ax.pcolormesh(
        RG,
        ZG,
        speed,
        shading="auto",
        cmap=cmap,
        norm=norm,
        antialiased=False,
    )
    _add_subsampled_quiver(ax, RG, ZG, u, w)

    zi, ri = np.where(edges)
    ec = np.zeros((len(zi), 4), dtype=np.float64)
    for k in range(len(zi)):
        lb = labels[zi[k], ri[k]]
        if np.isfinite(lb):
            idx = int(np.clip(lb, 0, n_reg - 1))
            ec[k] = region_colors[idx]
        else:
            ec[k] = (0.25, 0.25, 0.25, 1.0)
    ax.scatter(
        rcoord[ri],
        zcoord[zi],
        c=ec,
        s=(border_lw_pts * 2.2) ** 2 / 16.0,
        marker="s",
        linewidths=0,
        zorder=10,
        rasterized=True,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_xlim(float(np.min(rcoord)), float(np.max(rcoord)))
    z_min, z_max = float(np.min(zcoord)), float(np.max(zcoord))
    ax.set_ylim(z_min, z_max)
    if invert_z:
        ax.invert_yaxis()

    display = field_label if field_label is not None else "|v| = √(U²+W²)"
    ttl = f"{stem} — {dataset.upper()} — {display}"
    ax.set_title(ttl, fontsize=11)

    cb = fig.colorbar(pcm, cax=cax, orientation="vertical")
    cb.set_label(display)

    leg_handles = [
        mpatches.Patch(
            color=tuple(region_colors[i, :3]),
            label=name,
            alpha=float(region_colors[i, 3]),
        )
        for i, name in enumerate(mask_names)
    ]
    leg_ax.legend(
        handles=leg_handles,
        loc="center left",
        fontsize=8,
        framealpha=0.9,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.08)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_snapshot_all(
    data: np.lib.npyio.NpzFile,
    dataset: str,
    masks: list[tuple[str, str]],
    limits_by_target: dict[str, tuple[float, float, str]],
    invert_z: bool,
    stem: str,
    out_png: str,
    mirror_r: bool = True,
    border_lw_pts: float = 2.5,
) -> None:
    """One row: velocity (|v| + quiver), av_density, av_pressure.

    Fixed ``subplots_adjust`` and shared axes; save without ``bbox_inches='tight'``
    for stable GIF geometry.
    """
    for t in ALL_PANEL_TARGETS:
        if t == "velocity":
            if "Uvelocity" not in data or "Wvelocity" not in data:
                sys.exit("-t all requires Uvelocity and Wvelocity.")
        elif t not in data:
            sys.exit(f"-t all requires array {t!r}; missing in this snapshot.")

    _, mask_names = build_component_labels(data, masks)
    n_reg = len(mask_names)
    region_colors = np.array(
        [plt.cm.tab10(i / max(n_reg - 1, 1) * 0.95) for i in range(n_reg)],
        dtype=np.float64,
    )
    leg_handles = [
        mpatches.Patch(
            color=tuple(region_colors[i, :3]),
            label=name,
            alpha=float(region_colors[i, 3]),
        )
        for i, name in enumerate(mask_names)
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.5, 6.5),
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.76,
        top=0.88,
        bottom=0.14,
        wspace=0.34,
    )

    for ax, panel_key in zip(np.asarray(axes).flat, ALL_PANEL_TARGETS):
        vmin, vmax, field_label = limits_by_target[panel_key]
        if panel_key == "velocity":
            zcoord, rcoord, u, w, labels = aligned_u_w_velocity(data, dataset, masks)
            if mirror_r:
                rcoord, u, w, labels = mirror_vector_r_half_slice(rcoord, u, w, labels)
            speed = np.sqrt(u * u + w * w)
            RG, ZG = np.meshgrid(rcoord, zcoord)
            cmap, norm = colormap_and_norm(vmin, vmax)
            pcm = ax.pcolormesh(
                RG,
                ZG,
                speed,
                shading="auto",
                cmap=cmap,
                norm=norm,
                antialiased=False,
            )
            _add_subsampled_quiver(ax, RG, ZG, u, w)
            edges = edges_from_labels(labels)
        else:
            fld = np.nan_to_num(data[panel_key].astype(np.float64))
            rcoord = np.asarray(data["Rcoord"], dtype=np.float64)
            zcoord = np.asarray(data["Zcoord"], dtype=np.float64)
            if fld.shape != (len(zcoord), len(rcoord)):
                sys.exit(
                    f"Shape mismatch: {panel_key} is {fld.shape}, "
                    f"expected ({len(zcoord)}, {len(rcoord)}) from Zcoord × Rcoord."
                )
            labels, _ = build_component_labels(data, masks)
            zcoord, fld, labels = _align_z_vertical(zcoord, fld, labels, dataset)
            if mirror_r:
                rcoord, fld, labels = mirror_r_half_slice(rcoord, fld, labels)
            edges = edges_from_labels(labels)
            RG, ZG = np.meshgrid(rcoord, zcoord)
            cmap, norm = colormap_and_norm(vmin, vmax)
            pcm = ax.pcolormesh(
                RG,
                ZG,
                fld,
                shading="auto",
                cmap=cmap,
                norm=norm,
                antialiased=False,
            )

        zi, ri = np.where(edges)
        ec = np.zeros((len(zi), 4), dtype=np.float64)
        for k in range(len(zi)):
            lb = labels[zi[k], ri[k]]
            if np.isfinite(lb):
                idx = int(np.clip(lb, 0, n_reg - 1))
                ec[k] = region_colors[idx]
            else:
                ec[k] = (0.25, 0.25, 0.25, 1.0)
        ax.scatter(
            rcoord[ri],
            zcoord[zi],
            c=ec,
            s=(border_lw_pts * 2.2) ** 2 / 16.0,
            marker="s",
            linewidths=0,
            zorder=10,
            rasterized=True,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(np.min(rcoord)), float(np.max(rcoord)))
        z_min, z_max = float(np.min(zcoord)), float(np.max(zcoord))
        ax.set_ylim(z_min, z_max)
        cb = fig.colorbar(
            pcm,
            ax=ax,
            fraction=0.085,
            pad=0.025,
        )
        cb.set_label(field_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(field_label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("R", fontsize=10)

    for ax in np.asarray(axes).flat:
        if invert_z:
            ax.invert_yaxis()

    np.ravel(axes)[0].set_ylabel("Z", fontsize=10)

    fig.legend(
        handles=leg_handles,
        loc="center left",
        bbox_to_anchor=(0.785, 0.52),
        fontsize=8,
        framealpha=0.9,
        borderaxespad=0,
    )
    fig.suptitle(
        f"{stem} — {dataset.upper()} — velocity | density | pressure",
        fontsize=12,
        y=0.97,
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def make_gif(frame_dir: str, out_gif: str, duration_ms: int) -> bool:
    """Return True if a GIF was written."""
    pngs = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if len(pngs) < 2:
        return False
    frames = [Image.open(p) for p in pngs]
    first, *rest = frames
    first.save(
        out_gif,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )
    for f in frames:
        f.close()
    print(f"  gif ({len(pngs)} frames) → {out_gif}")
    return True


def delete_pngs_in_dir(frame_dir: str) -> int:
    """Remove all ``*.png`` in *frame_dir*; return how many were deleted."""
    n = 0
    for p in glob.glob(os.path.join(frame_dir, "*.png")):
        try:
            os.remove(p)
            n += 1
        except OSError as e:
            print(f"  warning: could not remove {p}: {e}", file=sys.stderr)
    if n:
        print(f"  removed {n} PNG file(s) from {frame_dir}")
    return n


def invert_z_resolve(invert_flag: bool, no_invert_flag: bool) -> bool:
    """Optional matplotlib y-axis invert on top of Z reflection (normally off)."""
    if invert_flag and no_invert_flag:
        sys.exit("Use only one of --invert-z and --no-invert-z")
    if invert_flag:
        return True
    if no_invert_flag:
        return False
    return False


def parse_frame_policy(policy: str | None) -> str:
    pol = (policy or "even").lower()
    if pol not in ("even", "first"):
        sys.exit("--frame-policy must be 'even' or 'first'")
    return pol


# ── CYL ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HEAT visualization: one target field colors the grid; VoF regions + borders."
    )
    parser.add_argument(
        "dir",
        help="Directory containing .npz snapshots (e.g. data/cyl/id00001)",
    )
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        metavar="FIELD|all|velocity",
        help="npz field for the color gradient; 'all' = one row: velocity (|v|+quiver), "
        "density, pressure; 'velocity' = √(U²+W²) with direction arrows only.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("cyl", "pli"),
        default=None,
        help="Dataset kind (default: infer from .../data/cyl/ or .../data/pli/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory under <project>/out/ (default: out/<dataset>/<id>_vis)",
    )
    parser.add_argument("--frames", type=int, default=None, help="Max .npz snapshots to use")
    parser.add_argument(
        "--frame-policy",
        choices=("even", "first"),
        default=None,
        help="How to subsample when using --frames (default: even)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="GIF frame rate when building animation (default: 5)",
    )
    parser.add_argument(
        "--invert-z",
        action="store_true",
        help="Extra matplotlib y inversion after Z reflection (usually unnecessary)",
    )
    parser.add_argument(
        "--no-invert-z",
        action="store_true",
        help="Same as default (explicit); no extra y inversion",
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="Only build a GIF from PNGs already in DIR (must be under out/)",
    )
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    parser.add_argument(
        "--cleanup-pngs",
        action="store_true",
        help="After a GIF is created, delete the frame PNGs in the output directory",
    )
    parser.add_argument(
        "--scale",
        choices=("physics", "percentile"),
        default="physics",
        help="Color limits: physics uses documented vmax per known field; "
        "percentile uses first-frame 2–98%% for any target.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not reflect the R half-slice across R=0 (keep native R range only).",
    )
    args = parser.parse_args()

    dataset = args.dataset or detect_dataset_kind(args.dir)
    if not dataset:
        sys.exit(
            "Could not infer --dataset; pass -d cyl or -d pli, "
            "or use a path under .../data/cyl/... or .../data/pli/..."
        )

    masks = COMPONENT_MASKS_CYL if dataset == "cyl" else COMPONENT_MASKS_PLI
    is_all = args.target.strip().lower() == "all"
    is_velocity = args.target.strip().lower() == "velocity"
    if is_all:
        target = "all"
    elif is_velocity:
        target = "velocity"
    else:
        target = args.target

    fps = float(args.fps)
    duration_ms = int(round(1000 / max(fps, 1e-6)))

    base_out = resolve_output_under_out(os.path.abspath(args.dir), dataset, args.output)

    if args.gif_only:
        assert_under_out(os.path.abspath(args.dir))
        if args.cleanup_pngs and args.no_gif:
            sys.exit("--cleanup-pngs does not apply with --no-gif (no GIF is built).")
        gif_out = os.path.join(os.path.abspath(args.dir), "animation.gif")
        ok = make_gif(os.path.abspath(args.dir), gif_out, duration_ms=duration_ms)
        if args.cleanup_pngs and ok:
            delete_pngs_in_dir(os.path.abspath(args.dir))
        elif args.cleanup_pngs and not ok:
            print("  cleanup-pngs: no GIF was created (need at least 2 PNGs); leaving files.", file=sys.stderr)
        print("Done.")
        return

    if not os.path.isdir(args.dir):
        sys.exit(f"Not a directory: {args.dir}")

    all_npz = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    if not all_npz:
        sys.exit(f"No .npz files in {args.dir}")

    policy = parse_frame_policy(args.frame_policy)
    npz_paths = select_npz_files(all_npz, args.frames, policy)

    inv_z = invert_z_resolve(args.invert_z, args.no_invert_z)

    print(f"Dataset {dataset}  target={args.target}  invert_z={inv_z}  mirror_r={not args.no_mirror}")
    print(f"Using {len(npz_paths)} of {len(all_npz)} frames  policy={policy}")
    if is_all:
        probe = np.load(npz_paths[0])
        try:
            for t in ALL_PANEL_TARGETS:
                if t == "velocity":
                    if "Uvelocity" not in probe or "Wvelocity" not in probe:
                        sys.exit("-t all requires Uvelocity and Wvelocity in snapshots.")
                elif t not in probe:
                    sys.exit(
                        f"-t all: first snapshot missing required array {t!r}."
                    )
        finally:
            probe.close()

    if is_velocity:
        probe = np.load(npz_paths[0])
        try:
            if "Uvelocity" not in probe or "Wvelocity" not in probe:
                sys.exit("-t velocity requires Uvelocity and Wvelocity in snapshots.")
        finally:
            probe.close()

    panel_out = os.path.join(base_out, ALL_PANEL_SUBDIR) if is_all else base_out
    os.makedirs(base_out, exist_ok=True)
    if is_all:
        os.makedirs(panel_out, exist_ok=True)

    if is_all:
        limits_by_target: dict[str, tuple[float, float, str]] = {}
        for t in ALL_PANEL_TARGETS:
            lim_vmin, lim_vmax, flab = resolve_animation_color_limits(
                t, dataset, masks, npz_paths, args.scale
            )
            limits_by_target[t] = (lim_vmin, lim_vmax, flab)
        print(f"Writing PNGs under {panel_out}\n")
        print("Color scales (fixed for whole run):")
        for t in ALL_PANEL_TARGETS:
            a, b, lab = limits_by_target[t]
            ph = args.scale == "physics" and t in TARGET_FIELD_PHYSICS
            src = (
                "physics defaults"
                if ph
                else f"first frame ({os.path.basename(npz_paths[0])})"
            )
            print(f"  {t} ({src}): vmin={a:.6g}  vmax={b:.6g}  — {lab}")
        print()
    else:
        lim_vmin, lim_vmax, field_label = resolve_animation_color_limits(
            target, dataset, masks, npz_paths, args.scale
        )
        physics_used = args.scale == "physics" and target in TARGET_FIELD_PHYSICS
        scale_src = (
            "physics defaults"
            if physics_used
            else f"first frame ({os.path.basename(npz_paths[0])})"
        )
        print(f"Writing PNGs under {base_out}\n")
        print(
            f"Color scale ({scale_src}): vmin={lim_vmin:.6g}  vmax={lim_vmax:.6g}  — {field_label}\n"
        )

    for path in npz_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        data = np.load(path)
        if is_all:
            plot_snapshot_all(
                data,
                dataset,
                masks,
                limits_by_target,
                invert_z=inv_z,
                stem=stem,
                out_png=os.path.join(panel_out, f"{stem}.png"),
                mirror_r=not args.no_mirror,
            )
        elif is_velocity:
            plot_snapshot_velocity(
                data,
                dataset,
                masks,
                invert_z=inv_z,
                stem=stem,
                out_png=os.path.join(base_out, f"{stem}.png"),
                vmin=lim_vmin,
                vmax=lim_vmax,
                field_label=field_label,
                mirror_r=not args.no_mirror,
            )
        else:
            plot_snapshot(
                data,
                dataset,
                masks,
                target,
                invert_z=inv_z,
                stem=stem,
                out_png=os.path.join(base_out, f"{stem}.png"),
                vmin=lim_vmin,
                vmax=lim_vmax,
                field_label=field_label,
                mirror_r=not args.no_mirror,
            )
        data.close()

    if args.cleanup_pngs and args.no_gif:
        sys.exit("--cleanup-pngs requires GIF generation (omit --no-gif).")

    if not args.no_gif:
        gif_dir = panel_out if is_all else base_out
        if is_all:
            gif_name = "all.gif"
        elif is_velocity:
            gif_name = "velocity.gif"
        else:
            gif_name = f"{target.replace('/', '_')}.gif"
        gif_path = os.path.join(gif_dir, gif_name)
        ok = make_gif(gif_dir, gif_path, duration_ms=duration_ms)
        if args.cleanup_pngs and ok:
            delete_pngs_in_dir(gif_dir)
        elif args.cleanup_pngs and not ok:
            print(
                "  cleanup-pngs: no GIF was created (need at least 2 PNGs); leaving files.",
                file=sys.stderr,
            )

    print("Done.")


if __name__ == "__main__":
    main()
