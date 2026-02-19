#!/usr/bin/env python3
"""
adipose_analysis.py - Visceral and subcutaneous adipose tissue analysis

This module handles:
1. Extracting VAT from TotalSegmentator labels
2. Checking SAT FOV adequacy (truncation detection)
3. Computing adipose metrics within L1-L2 region
4. Calculating sarcopenic obesity indices
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)


# HU range for adipose tissue validation
FAT_HU_MIN = -190
FAT_HU_MAX = -30

# QC threshold: if more than this fraction of labeled voxels are outside fat HU range, flag as error
FAT_OUTLIER_THRESHOLD = 0.10

# Margin from image edge for SAT FOV check
SAT_EDGE_MARGIN_VOXELS = 5


@dataclass
class AdiposeAnalysisResult:
    """Results from adipose tissue analysis."""
    success: bool

    # VAT metrics (always computed)
    vat_volume_cm3: float
    vat_area_L1L2_cm2: float
    vat_hu_mean: float
    vat_hu_median: float
    vat_outlier_fraction: float

    # SAT metrics (conditional on FOV adequacy)
    sat_fov_adequate: bool
    sat_volume_cm3: Optional[float]
    sat_area_L1L2_cm2: Optional[float]
    sat_hu_mean: Optional[float]
    sat_hu_median: Optional[float]

    # Derived indices
    vat_sat_ratio: Optional[float]
    muscle_csa_vat_ratio: Optional[float]

    # Info
    z_range_used: Tuple[int, int]
    qc_messages: list


def find_adipose_masks(segmentations_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find VAT and SAT masks from TotalSegmentator output.

    Prioritizes tissue_4_types output (torso_fat, subcutaneous_fat) over legacy names.
    TotalSegmentator names vary by version, so we check multiple possibilities.

    Args:
        segmentations_dir: Directory containing TotalSegmentator outputs

    Returns:
        Tuple of (vat_path, sat_path) - may be None if not found
    """
    # Possible VAT label names (tissue_4_types output first, then legacy)
    vat_candidates = [
        "torso_fat.nii.gz",           # tissue_4_types output (preferred)
        "fat_visceral.nii.gz",
        "visceral_fat.nii.gz",
        "body_trunc.nii.gz",
    ]

    # Possible SAT label names (tissue_4_types output first, then legacy)
    sat_candidates = [
        "subcutaneous_fat.nii.gz",    # tissue_4_types output (preferred)
        "fat_subcutaneous.nii.gz",
    ]

    vat_path = None
    sat_path = None

    for name in vat_candidates:
        path = segmentations_dir / name
        if path.exists():
            vat_path = path
            logger.info(f"Found VAT mask: {name}")
            break

    for name in sat_candidates:
        path = segmentations_dir / name
        if path.exists():
            sat_path = path
            logger.info(f"Found SAT mask: {name}")
            break

    return vat_path, sat_path


def find_tissue_masks(segmentations_dir: Path) -> dict:
    """
    Find all tissue_4_types masks from TotalSegmentator output.

    Returns:
        Dictionary with paths to found masks:
        - 'vat': torso_fat (visceral adipose)
        - 'sat': subcutaneous_fat
        - 'muscle': skeletal_muscle
        - 'imat': intermuscular_fat
    """
    tissue_files = {
        "vat": "torso_fat.nii.gz",
        "sat": "subcutaneous_fat.nii.gz",
        "muscle": "skeletal_muscle.nii.gz",
        "imat": "intermuscular_fat.nii.gz"
    }

    found_masks = {}
    for key, filename in tissue_files.items():
        path = segmentations_dir / filename
        if path.exists():
            found_masks[key] = path
        else:
            found_masks[key] = None

    return found_masks


def check_sat_fov_adequate(sat_mask: np.ndarray,
                           z_range: Tuple[int, int],
                           margin_voxels: int = SAT_EDGE_MARGIN_VOXELS) -> bool:
    """
    Check if SAT is adequately captured (not truncated at image edges).

    Spine-focused FOV often cuts off lateral subcutaneous fat.
    If SAT touches the image boundary, it's truncated and measurements are unreliable.

    Args:
        sat_mask: SAT binary mask
        z_range: Slice range to check
        margin_voxels: How close to edge counts as "touching"

    Returns:
        True if SAT is fully captured, False if truncated
    """
    z_start, z_end = z_range

    for z in range(z_start, z_end + 1):
        if z >= sat_mask.shape[2]:
            continue

        slice_mask = sat_mask[:, :, z]
        if not slice_mask.any():
            continue

        # Check left edge (low x)
        if slice_mask[:margin_voxels, :].any():
            logger.info(f"SAT touches left edge at slice {z}")
            return False

        # Check right edge (high x)
        if slice_mask[-margin_voxels:, :].any():
            logger.info(f"SAT touches right edge at slice {z}")
            return False

        # Check front edge (low y)
        if slice_mask[:, :margin_voxels].any():
            logger.info(f"SAT touches front edge at slice {z}")
            return False

        # Check back edge (high y)
        if slice_mask[:, -margin_voxels:].any():
            logger.info(f"SAT touches back edge at slice {z}")
            return False

    return True


def compute_fat_metrics(ct_data: np.ndarray,
                        fat_mask: np.ndarray,
                        voxel_sizes: Tuple[float, float, float],
                        z_range: Tuple[int, int],
                        tissue_name: str) -> dict:
    """
    Compute volume and area metrics for a fat compartment.

    Args:
        ct_data: CT volume
        fat_mask: Binary mask of fat compartment
        voxel_sizes: Voxel dimensions in mm
        z_range: Slice range to analyze
        tissue_name: "VAT" or "SAT" for logging

    Returns:
        Dictionary with volume, area, HU stats, and outlier fraction
    """
    z_start, z_end = z_range

    # Restrict mask to z-range
    restricted_mask = np.zeros_like(fat_mask, dtype=bool)
    for z in range(z_start, min(z_end + 1, fat_mask.shape[2])):
        restricted_mask[:, :, z] = fat_mask[:, :, z]

    voxel_vol_cm3 = np.prod(voxel_sizes) / 1000.0
    voxel_area_cm2 = (voxel_sizes[0] * voxel_sizes[1]) / 100.0

    # Volume
    volume_cm3 = restricted_mask.sum() * voxel_vol_cm3

    # Cross-sectional area (average across slices)
    areas = []
    for z in range(z_start, min(z_end + 1, fat_mask.shape[2])):
        slice_mask = restricted_mask[:, :, z]
        if slice_mask.any():
            areas.append(slice_mask.sum() * voxel_area_cm2)

    area_cm2 = np.mean(areas) if areas else 0.0

    # HU statistics
    hu_values = ct_data[restricted_mask]

    if len(hu_values) > 0:
        hu_mean = float(np.mean(hu_values))
        hu_median = float(np.median(hu_values))

        # Outlier fraction (voxels outside typical fat HU range)
        in_range = (hu_values >= FAT_HU_MIN) & (hu_values <= FAT_HU_MAX)
        outlier_fraction = 1.0 - (in_range.sum() / len(hu_values))
    else:
        hu_mean = np.nan
        hu_median = np.nan
        outlier_fraction = 1.0

    logger.info(f"{tissue_name}: volume={volume_cm3:.1f} cm³, area={area_cm2:.1f} cm², "
                f"HU={hu_mean:.1f}, outliers={outlier_fraction:.1%}")

    return {
        "volume_cm3": volume_cm3,
        "area_cm2": area_cm2,
        "hu_mean": hu_mean,
        "hu_median": hu_median,
        "outlier_fraction": outlier_fraction
    }


def analyze_adipose(ct_path: Path,
                    segmentations_dir: Path,
                    vertebral_bodies_dir: Path,
                    muscle_csa_cm2: Optional[float] = None,
                    skip_sat: bool = False) -> AdiposeAnalysisResult:
    """
    Run complete adipose tissue analysis.

    Args:
        ct_path: Path to CT NIfTI file
        segmentations_dir: Directory with TotalSegmentator outputs
        vertebral_bodies_dir: Directory with L1/L2 body masks (for z-range)
        muscle_csa_cm2: Muscle CSA for computing ratio (optional)
        skip_sat: If True, skip SAT analysis entirely

    Returns:
        AdiposeAnalysisResult with all metrics
    """
    qc_messages = []

    # Load CT
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()
    voxel_sizes = ct_nii.header.get_zooms()[:3]

    # Determine z-range from L1/L2 vertebrae
    l1_path = vertebral_bodies_dir / "L1_body.nii.gz"
    l2_path = vertebral_bodies_dir / "L2_body.nii.gz"

    if l1_path.exists() and l2_path.exists():
        l1_mask = nib.load(l1_path).get_fdata().astype(bool)
        l2_mask = nib.load(l2_path).get_fdata().astype(bool)

        l1_z = np.where(l1_mask.any(axis=(0, 1)))[0]
        l2_z = np.where(l2_mask.any(axis=(0, 1)))[0]

        if len(l1_z) > 0 and len(l2_z) > 0:
            z_start = min(l1_z.min(), l2_z.min())
            z_end = max(l1_z.max(), l2_z.max())
        else:
            qc_messages.append("WARNING: Could not determine z-range from vertebrae")
            z_start, z_end = 0, ct_data.shape[2] - 1
    else:
        qc_messages.append("WARNING: Vertebral body masks not found, using full z-range")
        z_start, z_end = 0, ct_data.shape[2] - 1

    z_range = (z_start, z_end)
    logger.info(f"Adipose analysis z-range: {z_start} to {z_end}")

    # Find adipose masks
    vat_path, sat_path = find_adipose_masks(segmentations_dir)

    # Initialize result with defaults
    result = AdiposeAnalysisResult(
        success=False,
        vat_volume_cm3=0.0,
        vat_area_L1L2_cm2=0.0,
        vat_hu_mean=np.nan,
        vat_hu_median=np.nan,
        vat_outlier_fraction=1.0,
        sat_fov_adequate=False,
        sat_volume_cm3=None,
        sat_area_L1L2_cm2=None,
        sat_hu_mean=None,
        sat_hu_median=None,
        vat_sat_ratio=None,
        muscle_csa_vat_ratio=None,
        z_range_used=z_range,
        qc_messages=qc_messages
    )

    # Analyze VAT
    if vat_path is not None:
        vat_mask = nib.load(vat_path).get_fdata().astype(bool)
        vat_metrics = compute_fat_metrics(ct_data, vat_mask, voxel_sizes, z_range, "VAT")

        result.vat_volume_cm3 = vat_metrics["volume_cm3"]
        result.vat_area_L1L2_cm2 = vat_metrics["area_cm2"]
        result.vat_hu_mean = vat_metrics["hu_mean"]
        result.vat_hu_median = vat_metrics["hu_median"]
        result.vat_outlier_fraction = vat_metrics["outlier_fraction"]

        if vat_metrics["outlier_fraction"] > FAT_OUTLIER_THRESHOLD:
            qc_messages.append(f"WARNING: VAT outlier fraction = {vat_metrics['outlier_fraction']:.1%} > 10%")

        result.success = True
    else:
        qc_messages.append("ERROR: VAT mask not found in TotalSegmentator output")
        logger.warning("VAT mask not found - may need to run TotalSegmentator with --task total")

    # Analyze SAT (conditional)
    if not skip_sat and sat_path is not None:
        sat_mask = nib.load(sat_path).get_fdata().astype(bool)

        # Check FOV adequacy
        sat_fov_ok = check_sat_fov_adequate(sat_mask, z_range)
        result.sat_fov_adequate = sat_fov_ok

        if sat_fov_ok:
            sat_metrics = compute_fat_metrics(ct_data, sat_mask, voxel_sizes, z_range, "SAT")

            result.sat_volume_cm3 = sat_metrics["volume_cm3"]
            result.sat_area_L1L2_cm2 = sat_metrics["area_cm2"]
            result.sat_hu_mean = sat_metrics["hu_mean"]
            result.sat_hu_median = sat_metrics["hu_median"]

            if sat_metrics["outlier_fraction"] > FAT_OUTLIER_THRESHOLD:
                qc_messages.append(f"WARNING: SAT outlier fraction = {sat_metrics['outlier_fraction']:.1%} > 10%")

            # VAT/SAT ratio
            if result.sat_volume_cm3 > 0:
                result.vat_sat_ratio = result.vat_volume_cm3 / result.sat_volume_cm3

        else:
            qc_messages.append("INFO: SAT FOV inadequate (truncated), SAT metrics excluded")
            logger.info("SAT appears truncated at image boundary, excluding from analysis")

    elif skip_sat:
        qc_messages.append("INFO: SAT analysis skipped by user request")
    elif sat_path is None:
        qc_messages.append("INFO: SAT mask not found in TotalSegmentator output")

    # Muscle CSA / VAT ratio (sarcopenic obesity index)
    if muscle_csa_cm2 is not None and result.vat_area_L1L2_cm2 > 0:
        result.muscle_csa_vat_ratio = muscle_csa_cm2 / result.vat_area_L1L2_cm2

    result.qc_messages = qc_messages
    return result


def _convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        val = float(obj)
        # Convert NaN to None for valid JSON
        return None if np.isnan(val) else val
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        # Handle Python float NaN
        return None if np.isnan(obj) else obj
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(v) for v in obj]
    return obj


def save_adipose_results(result: AdiposeAnalysisResult, output_path: Path):
    """Save adipose analysis results to JSON."""
    output_dict = asdict(result)
    # Convert z_range tuple to list for JSON
    output_dict["z_range_used"] = list(result.z_range_used)
    # Convert numpy types to native Python types
    output_dict = _convert_to_json_serializable(output_dict)

    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    logger.info(f"Adipose results saved to {output_path}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 4:
        print("Usage: python adipose_analysis.py <ct_path> <seg_dir> <vb_dir> [--skip-sat]")
        sys.exit(1)

    ct_path = Path(sys.argv[1])
    seg_dir = Path(sys.argv[2])
    vb_dir = Path(sys.argv[3])
    skip_sat = "--skip-sat" in sys.argv

    result = analyze_adipose(ct_path, seg_dir, vb_dir, skip_sat=skip_sat)

    print(f"\nAdipose Analysis {'successful' if result.success else 'failed'}")
    print(f"VAT volume: {result.vat_volume_cm3:.1f} cm³")
    print(f"VAT area: {result.vat_area_L1L2_cm2:.1f} cm²")
    print(f"VAT HU: {result.vat_hu_mean:.1f} (outliers: {result.vat_outlier_fraction:.1%})")
    print(f"SAT FOV adequate: {result.sat_fov_adequate}")
    if result.sat_volume_cm3 is not None:
        print(f"SAT volume: {result.sat_volume_cm3:.1f} cm³")
        print(f"VAT/SAT ratio: {result.vat_sat_ratio:.2f}")
    print("\nQC messages:")
    for msg in result.qc_messages:
        print(f"  {msg}")
