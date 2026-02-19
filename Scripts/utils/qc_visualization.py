#!/usr/bin/env python3
"""
qc_visualization.py - Quality control image generation

This module generates QC images for manual review:
1. Phantom detection and calibration
2. Vertebral body isolation
3. Bone segmentation
4. Muscle envelope and classification
5. Adipose tissue segmentation
6. Summary montage
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

# Color maps for different visualizations
TISSUE_COLORS = {
    "imat": "#FFD700",        # Gold
    "muscle_low": "#FFA500",  # Orange
    "muscle_normal": "#FF4500", # Red-orange
    "bone": "#FFFFFF",        # White
    "vat": "#00FF00",         # Green
    "sat": "#00FFFF",         # Cyan
}


def normalize_ct_window(ct_data: np.ndarray,
                        window_center: float = 40,
                        window_width: float = 400) -> np.ndarray:
    """Apply CT windowing for visualization."""
    low = window_center - window_width / 2
    high = window_center + window_width / 2
    normalized = np.clip(ct_data, low, high)
    normalized = (normalized - low) / (high - low)
    return normalized


def get_mid_slice(mask: np.ndarray, axis: int = 2) -> int:
    """Find middle slice containing the mask."""
    indices = np.where(mask.any(axis=tuple(i for i in range(3) if i != axis)))[0]
    if len(indices) == 0:
        return mask.shape[axis] // 2
    return indices[len(indices) // 2]


def create_phantom_detection_figure(ct_path: Path,
                                    calibration_dir: Path,
                                    output_path: Path):
    """
    Generate phantom detection QC image.

    Shows detected/clicked circles overlaid on phantom slice.
    Supports both manual calibration (phantom_calibration.json) and
    automated detection (calibration_bmd.json).
    """
    # Load CT
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()
    voxel_sizes = ct_nii.header.get_zooms()[:3]

    # Check for manual calibration first (preferred)
    manual_cal_path = calibration_dir / "phantom_calibration.json"
    rod_info_path = calibration_dir / "phantom_rois" / "phantom_rod_info.json"
    bmd_cal_path = calibration_dir / "calibration_bmd.json"

    # Try to load rod positions from various sources
    rod_positions = {}  # name -> (x, y, z, hu, density)
    method = "unknown"
    r_squared = 0.0
    z_mid = ct_data.shape[2] // 4  # Default to inferior quarter

    if manual_cal_path.exists():
        # Load manual calibration
        with open(manual_cal_path) as f:
            manual_cal = json.load(f)

        method = "manual"
        r_squared = manual_cal.get("calibration", {}).get("r_squared", 0.0)

        for rod in manual_cal.get("rods", []):
            center = rod.get("center_voxel", [0, 0, 0])
            rod_positions[rod["name"]] = {
                "x": center[0],
                "y": center[1],
                "z": center[2],
                "hu": rod["mean_hu"],
                "density": rod["density_mgcm3"]
            }

        if rod_positions:
            z_mid = list(rod_positions.values())[0]["z"]

    elif rod_info_path.exists():
        # Load from phantom_rois directory
        with open(rod_info_path) as f:
            rod_info = json.load(f)

        method = "automated"
        z_range = rod_info.get("z_range", [0, 30])
        z_mid = (z_range[0] + z_range[1]) // 2

        for name, data in rod_info.get("rods", {}).items():
            center = data.get("center_voxel", [0, 0, 0])
            rod_positions[name] = {
                "x": center[0],
                "y": center[1],
                "z": center[2] if len(center) > 2 else z_mid,
                "hu": data.get("measured_hu_mean", 0),
                "density": data.get("known_density_mgcm3", 0)
            }

        # Get R² from bmd calibration
        if bmd_cal_path.exists():
            with open(bmd_cal_path) as f:
                bmd_cal = json.load(f)
            r_squared = bmd_cal.get("regression", {}).get("r_squared", 0.0)

    elif bmd_cal_path.exists():
        # Fallback to basic calibration file
        with open(bmd_cal_path) as f:
            cal = json.load(f)

        method = cal.get("method", "automated")
        z_range = cal.get("phantom_z_range", [0, 30])
        z_mid = (z_range[0] + z_range[1]) // 2
        r_squared = cal.get("regression", {}).get("r_squared", 0.0)

        # No position info in basic calibration
        for name, data in cal.get("calibration_points", {}).items():
            rod_positions[name] = {
                "x": None,
                "y": None,
                "z": z_mid,
                "hu": data["hu"],
                "density": data["bmd_mgcm3"]
            }

    else:
        logger.warning("No calibration file found, skipping phantom QC image")
        return

    # Get phantom slice
    phantom_slice = ct_data[:, :, z_mid]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Display CT slice (NO transpose - consistent with manual_calibration.py)
    ax.imshow(phantom_slice, cmap='gray', origin='lower',
              vmin=-200, vmax=400)

    # Color scheme for rods
    rod_colors = {
        "fat": "purple",
        "base": "blue",
        "bone_50": "green",
        "bone_100": "orange",
        "bone_200": "red"
    }

    # Draw circles at rod positions (if available)
    # center_voxel in JSON is [data_i, data_j, z] where ct_data[data_i, data_j, z]
    # With no transpose: data_i = row (y), data_j = col (x)
    # So: display_x = data_j, display_y = data_i
    radius_px = 4.0 / np.mean(voxel_sizes[:2])  # 4mm radius

    for name, data in rod_positions.items():
        data_i, data_j = data.get("x"), data.get("y")  # Stored as [data_i, data_j]
        if data_i is not None and data_j is not None:
            color = rod_colors.get(name, "yellow")
            # Convert to display coordinates (no transpose)
            display_x = data_j  # col = x
            display_y = data_i  # row = y
            circle = Circle((display_x, display_y), radius=radius_px, color=color,
                           fill=False, linewidth=2)
            ax.add_patch(circle)

            # Label
            ax.text(display_x + radius_px + 3, display_y,
                   f"{name}\nHU={data['hu']:.1f}",
                   color=color, fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Title with method info
    method_str = "Manual Clicks" if method == "manual" else "Automated Detection"
    ax.set_title(f"Phantom Calibration ({method_str})\n"
                 f"Slice z={z_mid} | R² = {r_squared:.4f}")
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Y (voxels)")

    # Add text annotation for all rods
    y_offset = 0.98
    for name, data in sorted(rod_positions.items(), key=lambda x: x[1]["density"]):
        hu = data["hu"]
        density = data["density"]
        color = rod_colors.get(name, "white")
        ax.text(0.02, y_offset, f"{name}: HU={hu:.1f}, BMD={density} mg/cm³",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', color=color,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        y_offset -= 0.045

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved phantom detection figure: {output_path}")


def create_calibration_curve_figure(calibration_dir: Path,
                                    output_path: Path):
    """
    Generate calibration curve plot.

    Shows HU vs BMD regression with data points.
    """
    cal_path = calibration_dir / "calibration_bmd.json"
    if not cal_path.exists():
        logger.warning("Calibration file not found, skipping curve figure")
        return

    with open(cal_path) as f:
        cal = json.load(f)

    cal_points = cal.get("calibration_points", {})
    regression = cal.get("regression", {})

    # Extract data points
    hu_values = []
    bmd_values = []
    names = []

    for name, data in cal_points.items():
        hu_values.append(data["hu"])
        bmd_values.append(data["bmd_mgcm3"])
        names.append(name)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot data points
    ax.scatter(hu_values, bmd_values, s=100, c='blue', zorder=5)

    # Add labels
    for i, name in enumerate(names):
        ax.annotate(name, (hu_values[i], bmd_values[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Plot regression line
    slope = regression.get("slope", 1)
    intercept = regression.get("intercept", 0)
    r_squared = regression.get("r_squared", 0)

    x_range = np.linspace(min(hu_values) - 20, max(hu_values) + 20, 100)
    y_fit = slope * x_range + intercept
    ax.plot(x_range, y_fit, 'r-', linewidth=2, label=f'y = {slope:.4f}x + {intercept:.2f}')

    ax.set_xlabel("HU (Hounsfield Units)")
    ax.set_ylabel("BMD (mg/cm³)")
    ax.set_title(f"Phantom Calibration Curve\nR² = {r_squared:.4f}")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved calibration curve figure: {output_path}")


def create_vertebral_isolation_figure(segmentations_dir: Path,
                                      vertebral_bodies_dir: Path,
                                      ct_path: Path,
                                      output_path: Path):
    """
    Generate vertebral body isolation comparison.

    Shows original TotalSegmentator mask vs isolated body side by side.
    """
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, vertebra in enumerate(["L1", "L2"]):
        # Body path uses standardized L1/L2 names
        body_path = vertebral_bodies_dir / f"{vertebra}_body.nii.gz"

        # Try to find original TotalSegmentator mask from detection result
        detection_path = vertebral_bodies_dir / "vertebra_detection.json"
        orig_path = None
        if detection_path.exists():
            with open(detection_path) as f:
                detection = json.load(f)
            if vertebra.lower() in detection and detection[vertebra.lower()]:
                orig_label = detection[vertebra.lower()].get("original_label")
                if orig_label:
                    orig_path = segmentations_dir / f"{orig_label}.nii.gz"

        # Fallback: try direct vertebra name
        if orig_path is None or not orig_path.exists():
            orig_path = segmentations_dir / f"vertebrae_{vertebra}.nii.gz"

        if not orig_path.exists() or not body_path.exists():
            continue

        orig_mask = nib.load(orig_path).get_fdata().astype(bool)
        body_mask = nib.load(body_path).get_fdata().astype(bool)

        # Get middle slice
        z_mid = get_mid_slice(orig_mask)

        orig_slice = orig_mask[:, :, z_mid]
        body_slice = body_mask[:, :, z_mid]
        ct_slice = normalize_ct_window(ct_data[:, :, z_mid])

        # Original mask
        axes[row, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[row, 0].contour(orig_slice.T, colors='red', linewidths=1)
        axes[row, 0].set_title(f"{vertebra} - Original (TotalSegmentator)")

        # Isolated body
        axes[row, 1].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[row, 1].contour(body_slice.T, colors='lime', linewidths=1)
        axes[row, 1].set_title(f"{vertebra} - Isolated Body")

        # Comparison overlay
        axes[row, 2].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[row, 2].contour(orig_slice.T, colors='red', linewidths=1, linestyles='dashed')
        axes[row, 2].contour(body_slice.T, colors='lime', linewidths=1)
        axes[row, 2].set_title(f"{vertebra} - Comparison (red=orig, green=body)")

    for ax in axes.flat:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved vertebral isolation figure: {output_path}")


def create_bone_segmentation_figure(vertebral_bodies_dir: Path,
                                    ct_path: Path,
                                    output_path: Path):
    """
    Generate bone segmentation QC image.

    Shows trabecular ROI overlay.
    """
    from importlib import import_module
    bone_analysis = import_module("03_bone_analysis")
    get_trabecular_mask = bone_analysis.get_trabecular_mask

    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for col, vertebra in enumerate(["L1", "L2"]):
        body_path = vertebral_bodies_dir / f"{vertebra}_body.nii.gz"

        if not body_path.exists():
            continue

        body_nii = nib.load(body_path)
        body_mask = body_nii.get_fdata().astype(bool)

        # Get trabecular mask
        trabecular_nii = get_trabecular_mask(body_nii)
        trabecular_mask = trabecular_nii.get_fdata().astype(bool)

        # Get middle slice
        z_mid = get_mid_slice(body_mask)

        body_slice = body_mask[:, :, z_mid]
        trab_slice = trabecular_mask[:, :, z_mid]
        ct_slice = normalize_ct_window(ct_data[:, :, z_mid])

        axes[col].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[col].contour(body_slice.T, colors='blue', linewidths=1, linestyles='dashed')
        axes[col].contour(trab_slice.T, colors='yellow', linewidths=2)
        axes[col].set_title(f"{vertebra} Trabecular ROI\n(blue=body, yellow=trabecular)")
        axes[col].set_xlabel("X")
        axes[col].set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved bone segmentation figure: {output_path}")


def create_muscle_envelope_figure(segmentations_dir: Path,
                                  envelope_path: Path,
                                  ct_path: Path,
                                  output_path: Path):
    """
    Generate muscle envelope QC image.

    Shows "Swiss cheese" original vs filled envelope.
    """
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()

    envelope_nii = nib.load(envelope_path)
    envelope_mask = envelope_nii.get_fdata().astype(bool)

    # Load original erector spinae masks - try multiple naming conventions
    muscle_names = [
        ("erector_spinae_left.nii.gz", "erector_spinae_right.nii.gz"),
        ("autochthon_left.nii.gz", "autochthon_right.nii.gz"),
    ]

    left_path = None
    right_path = None
    for left_name, right_name in muscle_names:
        l_path = segmentations_dir / left_name
        r_path = segmentations_dir / right_name
        if l_path.exists() and r_path.exists():
            left_path = l_path
            right_path = r_path
            break

    if left_path is None or right_path is None:
        logger.warning("Erector spinae/autochthon masks not found for envelope figure")
        return

    left_mask = nib.load(left_path).get_fdata().astype(bool)
    right_mask = nib.load(right_path).get_fdata().astype(bool)
    original = left_mask | right_mask

    # Get middle slice
    z_mid = get_mid_slice(envelope_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ct_slice = normalize_ct_window(ct_data[:, :, z_mid])
    orig_slice = original[:, :, z_mid]
    env_slice = envelope_mask[:, :, z_mid]

    # Original ("Swiss cheese")
    axes[0].imshow(ct_slice.T, cmap='gray', origin='lower')
    overlay = np.zeros((*ct_slice.shape, 4))
    overlay[orig_slice, :] = [1, 0, 0, 0.5]
    axes[0].imshow(overlay.transpose(1, 0, 2), origin='lower')
    axes[0].set_title("Original Erector Spinae\n(Swiss cheese pattern)")

    # Filled envelope
    axes[1].imshow(ct_slice.T, cmap='gray', origin='lower')
    overlay = np.zeros((*ct_slice.shape, 4))
    overlay[env_slice, :] = [0, 1, 0, 0.5]
    axes[1].imshow(overlay.transpose(1, 0, 2), origin='lower')
    axes[1].set_title("Filled Compartment Envelope\n(includes IMAT spaces)")

    # Comparison
    axes[2].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[2].contour(orig_slice.T, colors='red', linewidths=1, linestyles='dashed')
    axes[2].contour(env_slice.T, colors='lime', linewidths=2)
    axes[2].set_title("Comparison\n(red=original, green=envelope)")

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved muscle envelope figure: {output_path}")


def create_muscle_classification_figure(ct_path: Path,
                                        envelope_path: Path,
                                        calibration_dir: Path,
                                        output_path: Path):
    """
    Generate muscle tissue classification QC image.

    Color-coded voxels by tissue type.
    """
    from importlib import import_module
    muscle_analysis = import_module("04_muscle_analysis")
    classify_voxels = muscle_analysis.classify_voxels
    THRESHOLDS = muscle_analysis.THRESHOLDS

    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()

    envelope_nii = nib.load(envelope_path)
    envelope_mask = envelope_nii.get_fdata().astype(bool)

    # Load drift correction
    drift_offset = 0.0
    stability_path = calibration_dir / "calibration_hu_stability.json"
    if stability_path.exists():
        with open(stability_path) as f:
            stability = json.load(f)
        drift_offset = stability.get("drift_correction", {}).get("offset_hu", 0.0)

    # Classify voxels
    tissue_masks = classify_voxels(ct_data, envelope_mask, drift_offset)

    # Get middle slice
    z_mid = get_mid_slice(envelope_mask)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ct_slice = normalize_ct_window(ct_data[:, :, z_mid])

    # CT background
    axes[0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[1].imshow(ct_slice.T, cmap='gray', origin='lower')

    # Create classification overlay
    classification = np.zeros((*ct_slice.shape, 4))

    if tissue_masks["imat"][:, :, z_mid].any():
        classification[tissue_masks["imat"][:, :, z_mid], :] = [1, 0.84, 0, 0.7]  # Gold

    if tissue_masks["muscle_low"][:, :, z_mid].any():
        classification[tissue_masks["muscle_low"][:, :, z_mid], :] = [1, 0.65, 0, 0.7]  # Orange

    if tissue_masks["muscle_normal"][:, :, z_mid].any():
        classification[tissue_masks["muscle_normal"][:, :, z_mid], :] = [1, 0, 0, 0.7]  # Red

    axes[1].imshow(classification.transpose(1, 0, 2), origin='lower')

    # Add legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor='gold', alpha=0.7, label='IMAT (-190 to -30 HU)'),
        plt.Rectangle((0,0), 1, 1, facecolor='orange', alpha=0.7, label='Low density (-29 to +29 HU)'),
        plt.Rectangle((0,0), 1, 1, facecolor='red', alpha=0.7, label='Normal muscle (+30 to +150 HU)')
    ]
    axes[1].legend(handles=legend_elements, loc='upper right')

    axes[0].set_title(f"CT at z={z_mid}\n(drift corrected by {drift_offset:.1f} HU)")
    axes[1].set_title("Tissue Classification")

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.contour(envelope_mask[:, :, z_mid].T, colors='white', linewidths=1, linestyles='dashed')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved muscle classification figure: {output_path}")


def create_vat_segmentation_figure(ct_path: Path,
                                   segmentations_dir: Path,
                                   vertebral_bodies_dir: Path,
                                   output_path: Path):
    """
    Generate VAT segmentation QC image.
    """
    from importlib import import_module
    adipose_analysis = import_module("05_adipose_analysis")
    find_adipose_masks = adipose_analysis.find_adipose_masks

    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()

    vat_path, sat_path = find_adipose_masks(segmentations_dir)

    if vat_path is None:
        logger.warning("VAT mask not found, skipping VAT figure")
        return

    vat_mask = nib.load(vat_path).get_fdata().astype(bool)

    # Get z-range from vertebrae
    l1_path = vertebral_bodies_dir / "L1_body.nii.gz"
    if l1_path.exists():
        l1_mask = nib.load(l1_path).get_fdata().astype(bool)
        z_mid = get_mid_slice(l1_mask)
    else:
        z_mid = get_mid_slice(vat_mask)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ct_slice = normalize_ct_window(ct_data[:, :, z_mid], window_center=0, window_width=500)

    # CT only
    axes[0].imshow(ct_slice.T, cmap='gray', origin='lower')
    axes[0].set_title(f"CT at z={z_mid} (fat window)")

    # VAT overlay
    axes[1].imshow(ct_slice.T, cmap='gray', origin='lower')
    overlay = np.zeros((*ct_slice.shape, 4))
    if vat_mask[:, :, z_mid].any():
        overlay[vat_mask[:, :, z_mid], :] = [0, 1, 0, 0.5]
    axes[1].imshow(overlay.transpose(1, 0, 2), origin='lower')
    axes[1].set_title("VAT Segmentation (green)")

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved VAT segmentation figure: {output_path}")


def create_sat_fov_check_figure(ct_path: Path,
                                segmentations_dir: Path,
                                output_path: Path):
    """
    Generate SAT FOV adequacy check image.
    """
    from importlib import import_module
    adipose_analysis = import_module("05_adipose_analysis")
    find_adipose_masks = adipose_analysis.find_adipose_masks

    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()

    vat_path, sat_path = find_adipose_masks(segmentations_dir)

    if sat_path is None:
        logger.warning("SAT mask not found, skipping SAT FOV figure")
        return

    sat_mask = nib.load(sat_path).get_fdata().astype(bool)
    z_mid = get_mid_slice(sat_mask)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ct_slice = normalize_ct_window(ct_data[:, :, z_mid], window_center=0, window_width=500)

    ax.imshow(ct_slice.T, cmap='gray', origin='lower')

    # SAT overlay
    overlay = np.zeros((*ct_slice.shape, 4))
    if sat_mask[:, :, z_mid].any():
        overlay[sat_mask[:, :, z_mid], :] = [0, 1, 1, 0.5]
    ax.imshow(overlay.transpose(1, 0, 2), origin='lower')

    # Mark image boundaries
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Edge margin')
    ax.axvline(x=ct_slice.shape[0]-5, color='red', linestyle='--', linewidth=2)
    ax.axhline(y=5, color='red', linestyle='--', linewidth=2)
    ax.axhline(y=ct_slice.shape[1]-5, color='red', linestyle='--', linewidth=2)

    ax.set_title(f"SAT FOV Check (z={z_mid})\nCyan=SAT, Red dashed=edge margin")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SAT FOV check figure: {output_path}")


def create_summary_montage(qc_dir: Path, output_path: Path):
    """
    Generate single-page summary montage from existing QC images.
    """
    expected_images = [
        "calibration_curve.png",
        "vertebral_body_isolation.png",
        "bone_segmentation.png",
        "muscle_envelope.png",
        "muscle_classification.png",
        "vat_segmentation.png"
    ]

    available = [qc_dir / img for img in expected_images if (qc_dir / img).exists()]

    if not available:
        logger.warning("No QC images found for montage")
        return

    n_images = len(available)
    cols = 3
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = np.array(axes).flatten()

    for i, img_path in enumerate(available):
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(img_path.stem)
        axes[i].axis('off')

    # Hide unused axes
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved summary montage: {output_path}")


def generate_all_qc_images(ct_path: Path,
                           derived_dir: Path,
                           qc_dir: Path):
    """
    Generate all QC images for a single scan.

    Args:
        ct_path: Path to CT NIfTI file
        derived_dir: Directory with derived data (segmentations, calibration, etc.)
        qc_dir: Output directory for QC images
    """
    qc_dir.mkdir(parents=True, exist_ok=True)

    segmentations_dir = derived_dir / "segmentations"
    vertebral_bodies_dir = derived_dir / "vertebral_bodies"
    calibration_dir = derived_dir

    envelope_path = derived_dir / "muscle_compartment.nii.gz"

    # Generate each QC image
    try:
        create_phantom_detection_figure(ct_path, calibration_dir,
                                        qc_dir / "phantom_detection.png")
    except Exception as e:
        logger.error(f"Failed to create phantom detection figure: {e}")

    try:
        create_calibration_curve_figure(calibration_dir,
                                        qc_dir / "calibration_curve.png")
    except Exception as e:
        logger.error(f"Failed to create calibration curve figure: {e}")

    try:
        create_vertebral_isolation_figure(segmentations_dir, vertebral_bodies_dir, ct_path,
                                          qc_dir / "vertebral_body_isolation.png")
    except Exception as e:
        logger.error(f"Failed to create vertebral isolation figure: {e}")

    try:
        create_bone_segmentation_figure(vertebral_bodies_dir, ct_path,
                                        qc_dir / "bone_segmentation.png")
    except Exception as e:
        logger.error(f"Failed to create bone segmentation figure: {e}")

    if envelope_path.exists():
        try:
            create_muscle_envelope_figure(segmentations_dir, envelope_path, ct_path,
                                          qc_dir / "muscle_envelope.png")
        except Exception as e:
            logger.error(f"Failed to create muscle envelope figure: {e}")

        try:
            create_muscle_classification_figure(ct_path, envelope_path, calibration_dir,
                                                qc_dir / "muscle_classification.png")
        except Exception as e:
            logger.error(f"Failed to create muscle classification figure: {e}")

    try:
        create_vat_segmentation_figure(ct_path, segmentations_dir, vertebral_bodies_dir,
                                       qc_dir / "vat_segmentation.png")
    except Exception as e:
        logger.error(f"Failed to create VAT segmentation figure: {e}")

    try:
        create_sat_fov_check_figure(ct_path, segmentations_dir,
                                    qc_dir / "sat_fov_check.png")
    except Exception as e:
        logger.error(f"Failed to create SAT FOV check figure: {e}")

    # Summary montage
    try:
        create_summary_montage(qc_dir, qc_dir / "summary_montage.png")
    except Exception as e:
        logger.error(f"Failed to create summary montage: {e}")

    logger.info(f"QC images generated in {qc_dir}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 4:
        print("Usage: python qc_visualization.py <ct_path> <derived_dir> <qc_dir>")
        sys.exit(1)

    ct_path = Path(sys.argv[1])
    derived_dir = Path(sys.argv[2])
    qc_dir = Path(sys.argv[3])

    generate_all_qc_images(ct_path, derived_dir, qc_dir)
