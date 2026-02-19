#!/usr/bin/env python3
"""
bone_analysis.py - Trabecular BMD measurements from L1/L2 vertebrae

This module handles:
1. Eroding vertebral body masks to isolate trabecular bone
2. Extracting HU values from trabecular regions
3. Converting HU to BMD using phantom calibration
4. Computing BMD metrics (mean, median, volume)
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


@dataclass
class VertebralBMDResult:
    """BMD results for a single vertebra."""
    vertebra: str
    vBMD_mean_mgcm3: float
    vBMD_median_mgcm3: float
    vBMD_std_mgcm3: float
    vBMD_P10_mgcm3: float
    vBMD_P90_mgcm3: float
    trabecular_volume_cm3: float
    total_body_volume_cm3: float
    voxel_count: int
    valid_voxel_count: int
    hu_mean: float
    hu_median: float


@dataclass
class BoneAnalysisResult:
    """Complete bone analysis results."""
    success: bool
    L1: Optional[VertebralBMDResult]
    L2: Optional[VertebralBMDResult]
    L1L2_vBMD_mean_mgcm3: Optional[float]
    L1L2_vBMD_weighted_mean_mgcm3: Optional[float]
    calibration_slope: float
    calibration_intercept: float
    erosion_mm: float
    qc_messages: list


# Valid HU range for trabecular bone
HU_MIN = -50
HU_MAX = 400

# Erosion distance for trabecular isolation
DEFAULT_EROSION_MM = 5.0


def erode_to_trabecular(mask: np.ndarray,
                        voxel_sizes: Tuple[float, float, float],
                        erosion_mm: float = DEFAULT_EROSION_MM) -> np.ndarray:
    """
    Erode vertebral body mask to isolate trabecular bone region.

    Uses distance transform to erode by a physical distance in mm.
    This removes cortical shell and leaves trabecular core.

    Args:
        mask: Binary mask of vertebral body
        voxel_sizes: Voxel dimensions in mm (x, y, z)
        erosion_mm: Erosion distance in mm

    Returns:
        Binary mask of trabecular region
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    # Distance transform gives distance from boundary in voxels
    # Scale by voxel size to get physical distance
    dist = distance_transform_edt(mask, sampling=voxel_sizes)

    # Keep only voxels more than erosion_mm from boundary
    trabecular = dist > erosion_mm

    return trabecular


def analyze_vertebra(body_nii: nib.Nifti1Image,
                     ct_nii: nib.Nifti1Image,
                     slope: float,
                     intercept: float,
                     vertebra_name: str,
                     erosion_mm: float = DEFAULT_EROSION_MM) -> VertebralBMDResult:
    """
    Analyze a single vertebra for BMD.

    Args:
        body_nii: Isolated vertebral body mask
        ct_nii: CT volume
        slope: Calibration slope (HU to mg/cm³)
        intercept: Calibration intercept
        vertebra_name: "L1" or "L2"
        erosion_mm: Erosion distance for trabecular isolation

    Returns:
        VertebralBMDResult with all metrics
    """
    body_mask = body_nii.get_fdata().astype(bool)
    ct_data = ct_nii.get_fdata()
    voxel_sizes = np.array(body_nii.header.get_zooms()[:3])

    # Calculate voxel volume in cm³
    voxel_vol_mm3 = np.prod(voxel_sizes)
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

    # Total body volume
    total_body_voxels = body_mask.sum()
    total_body_volume_cm3 = total_body_voxels * voxel_vol_cm3

    if total_body_voxels == 0:
        logger.warning(f"{vertebra_name}: Empty body mask")
        return VertebralBMDResult(
            vertebra=vertebra_name,
            vBMD_mean_mgcm3=np.nan,
            vBMD_median_mgcm3=np.nan,
            vBMD_std_mgcm3=np.nan,
            vBMD_P10_mgcm3=np.nan,
            vBMD_P90_mgcm3=np.nan,
            trabecular_volume_cm3=0.0,
            total_body_volume_cm3=0.0,
            voxel_count=0,
            valid_voxel_count=0,
            hu_mean=np.nan,
            hu_median=np.nan
        )

    # Erode to trabecular region
    trabecular_mask = erode_to_trabecular(body_mask, tuple(voxel_sizes), erosion_mm)
    trabecular_voxels = trabecular_mask.sum()

    if trabecular_voxels == 0:
        logger.warning(f"{vertebra_name}: No trabecular region after {erosion_mm}mm erosion, "
                       "trying 2mm")
        trabecular_mask = erode_to_trabecular(body_mask, tuple(voxel_sizes), 2.0)
        trabecular_voxels = trabecular_mask.sum()

        if trabecular_voxels == 0:
            logger.warning(f"{vertebra_name}: Still no trabecular region, using full body")
            trabecular_mask = body_mask
            trabecular_voxels = total_body_voxels

    # Extract HU values from trabecular region
    hu_values = ct_data[trabecular_mask]

    # Filter to valid HU range
    valid_mask = (hu_values >= HU_MIN) & (hu_values <= HU_MAX)
    valid_hu = hu_values[valid_mask]

    if len(valid_hu) == 0:
        logger.warning(f"{vertebra_name}: No valid HU values in range [{HU_MIN}, {HU_MAX}]")
        return VertebralBMDResult(
            vertebra=vertebra_name,
            vBMD_mean_mgcm3=np.nan,
            vBMD_median_mgcm3=np.nan,
            vBMD_std_mgcm3=np.nan,
            vBMD_P10_mgcm3=np.nan,
            vBMD_P90_mgcm3=np.nan,
            trabecular_volume_cm3=trabecular_voxels * voxel_vol_cm3,
            total_body_volume_cm3=total_body_volume_cm3,
            voxel_count=trabecular_voxels,
            valid_voxel_count=0,
            hu_mean=float(np.mean(hu_values)) if len(hu_values) > 0 else np.nan,
            hu_median=float(np.median(hu_values)) if len(hu_values) > 0 else np.nan
        )

    # Convert HU to BMD
    bmd_values = slope * valid_hu + intercept

    # Compute statistics
    result = VertebralBMDResult(
        vertebra=vertebra_name,
        vBMD_mean_mgcm3=float(np.mean(bmd_values)),
        vBMD_median_mgcm3=float(np.median(bmd_values)),
        vBMD_std_mgcm3=float(np.std(bmd_values)),
        vBMD_P10_mgcm3=float(np.percentile(bmd_values, 10)),
        vBMD_P90_mgcm3=float(np.percentile(bmd_values, 90)),
        trabecular_volume_cm3=len(valid_hu) * voxel_vol_cm3,
        total_body_volume_cm3=total_body_volume_cm3,
        voxel_count=trabecular_voxels,
        valid_voxel_count=len(valid_hu),
        hu_mean=float(np.mean(valid_hu)),
        hu_median=float(np.median(valid_hu))
    )

    logger.info(f"{vertebra_name}: vBMD mean={result.vBMD_mean_mgcm3:.1f} mg/cm³, "
                f"median={result.vBMD_median_mgcm3:.1f} mg/cm³, "
                f"volume={result.trabecular_volume_cm3:.2f} cm³")

    return result


def analyze_bone(ct_path: Path,
                 vertebral_bodies_dir: Path,
                 calibration_dir: Path,
                 erosion_mm: float = DEFAULT_EROSION_MM,
                 save_trabecular_masks: bool = True) -> BoneAnalysisResult:
    """
    Run complete bone analysis for L1 and L2 vertebrae.

    Args:
        ct_path: Path to CT NIfTI file
        vertebral_bodies_dir: Directory containing L1_body.nii.gz and L2_body.nii.gz
        calibration_dir: Directory containing calibration_bmd.json
        erosion_mm: Erosion distance for trabecular isolation

    Returns:
        BoneAnalysisResult with all metrics
    """
    qc_messages = []

    # Load calibration
    cal_path = calibration_dir / "calibration_bmd.json"
    if not cal_path.exists():
        logger.error(f"Calibration file not found: {cal_path}")
        return BoneAnalysisResult(
            success=False,
            L1=None,
            L2=None,
            L1L2_vBMD_mean_mgcm3=None,
            L1L2_vBMD_weighted_mean_mgcm3=None,
            calibration_slope=0.0,
            calibration_intercept=0.0,
            erosion_mm=erosion_mm,
            qc_messages=["Calibration file not found"]
        )

    with open(cal_path) as f:
        cal = json.load(f)

    slope = cal["regression"]["slope"]
    intercept = cal["regression"]["intercept"]
    r_squared = cal["regression"]["r_squared"]

    logger.info(f"Loaded calibration: slope={slope:.4f}, intercept={intercept:.2f}, R²={r_squared:.4f}")

    if r_squared < 0.95:
        qc_messages.append(f"WARNING: Calibration R² = {r_squared:.4f} < 0.95")

    # Load CT
    ct_nii = nib.load(ct_path)

    # Create trabecular masks directory
    trabecular_dir = vertebral_bodies_dir / "trabecular_masks"
    if save_trabecular_masks:
        trabecular_dir.mkdir(parents=True, exist_ok=True)

    # Analyze L1
    l1_path = vertebral_bodies_dir / "L1_body.nii.gz"
    l1_result = None
    if l1_path.exists():
        l1_nii = nib.load(l1_path)
        l1_result = analyze_vertebra(l1_nii, ct_nii, slope, intercept, "L1", erosion_mm)

        # Save trabecular mask for QC
        if save_trabecular_masks:
            l1_trabecular_nii = get_trabecular_mask(l1_nii, erosion_mm)
            nib.save(l1_trabecular_nii, trabecular_dir / "L1_trabecular.nii.gz")
            logger.info(f"Saved L1 trabecular mask to {trabecular_dir / 'L1_trabecular.nii.gz'}")

        if np.isnan(l1_result.vBMD_mean_mgcm3):
            qc_messages.append("WARNING: L1 BMD could not be calculated")
        elif l1_result.vBMD_mean_mgcm3 < 50 or l1_result.vBMD_mean_mgcm3 > 250:
            qc_messages.append(f"WARNING: L1 BMD = {l1_result.vBMD_mean_mgcm3:.1f} mg/cm³ "
                               "outside typical range (50-250)")
    else:
        qc_messages.append("ERROR: L1 body mask not found")

    # Analyze L2
    l2_path = vertebral_bodies_dir / "L2_body.nii.gz"
    l2_result = None
    if l2_path.exists():
        l2_nii = nib.load(l2_path)
        l2_result = analyze_vertebra(l2_nii, ct_nii, slope, intercept, "L2", erosion_mm)

        # Save trabecular mask for QC
        if save_trabecular_masks:
            l2_trabecular_nii = get_trabecular_mask(l2_nii, erosion_mm)
            nib.save(l2_trabecular_nii, trabecular_dir / "L2_trabecular.nii.gz")
            logger.info(f"Saved L2 trabecular mask to {trabecular_dir / 'L2_trabecular.nii.gz'}")

        if np.isnan(l2_result.vBMD_mean_mgcm3):
            qc_messages.append("WARNING: L2 BMD could not be calculated")
        elif l2_result.vBMD_mean_mgcm3 < 50 or l2_result.vBMD_mean_mgcm3 > 250:
            qc_messages.append(f"WARNING: L2 BMD = {l2_result.vBMD_mean_mgcm3:.1f} mg/cm³ "
                               "outside typical range (50-250)")
    else:
        qc_messages.append("ERROR: L2 body mask not found")

    # Calculate combined L1L2 metrics
    l1l2_mean = None
    l1l2_weighted_mean = None

    if l1_result and l2_result:
        if not np.isnan(l1_result.vBMD_mean_mgcm3) and not np.isnan(l2_result.vBMD_mean_mgcm3):
            # Simple average
            l1l2_mean = (l1_result.vBMD_mean_mgcm3 + l2_result.vBMD_mean_mgcm3) / 2

            # Volume-weighted average
            total_vol = l1_result.trabecular_volume_cm3 + l2_result.trabecular_volume_cm3
            if total_vol > 0:
                l1l2_weighted_mean = (
                    l1_result.vBMD_mean_mgcm3 * l1_result.trabecular_volume_cm3 +
                    l2_result.vBMD_mean_mgcm3 * l2_result.trabecular_volume_cm3
                ) / total_vol

            logger.info(f"L1L2 combined: mean={l1l2_mean:.1f} mg/cm³, "
                        f"weighted mean={l1l2_weighted_mean:.1f} mg/cm³")

    success = (l1_result is not None and l2_result is not None and
               not np.isnan(l1_result.vBMD_mean_mgcm3) and
               not np.isnan(l2_result.vBMD_mean_mgcm3))

    return BoneAnalysisResult(
        success=success,
        L1=l1_result,
        L2=l2_result,
        L1L2_vBMD_mean_mgcm3=l1l2_mean,
        L1L2_vBMD_weighted_mean_mgcm3=l1l2_weighted_mean,
        calibration_slope=slope,
        calibration_intercept=intercept,
        erosion_mm=erosion_mm,
        qc_messages=qc_messages
    )


def _convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_bone_results(result: BoneAnalysisResult, output_path: Path):
    """Save bone analysis results to JSON."""
    output_dict = {
        "success": result.success,
        "L1L2_vBMD_mean_mgcm3": result.L1L2_vBMD_mean_mgcm3,
        "L1L2_vBMD_weighted_mean_mgcm3": result.L1L2_vBMD_weighted_mean_mgcm3,
        "calibration": {
            "slope": result.calibration_slope,
            "intercept": result.calibration_intercept
        },
        "erosion_mm": result.erosion_mm,
        "qc_messages": result.qc_messages
    }

    if result.L1:
        output_dict["L1"] = asdict(result.L1)
    if result.L2:
        output_dict["L2"] = asdict(result.L2)

    # Convert numpy types to native Python types
    output_dict = _convert_to_json_serializable(output_dict)

    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    logger.info(f"Bone results saved to {output_path}")


def get_trabecular_mask(body_nii: nib.Nifti1Image,
                        erosion_mm: float = DEFAULT_EROSION_MM) -> nib.Nifti1Image:
    """
    Get trabecular mask for QC visualization.

    Args:
        body_nii: Vertebral body mask
        erosion_mm: Erosion distance

    Returns:
        NIfTI image of trabecular mask
    """
    body_mask = body_nii.get_fdata().astype(bool)
    voxel_sizes = body_nii.header.get_zooms()[:3]

    trabecular = erode_to_trabecular(body_mask, voxel_sizes, erosion_mm)

    return nib.Nifti1Image(trabecular.astype(np.uint8), body_nii.affine, body_nii.header)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 4:
        print("Usage: python bone_analysis.py <ct_path> <vertebral_bodies_dir> <calibration_dir>")
        sys.exit(1)

    ct_path = Path(sys.argv[1])
    vb_dir = Path(sys.argv[2])
    cal_dir = Path(sys.argv[3])

    result = analyze_bone(ct_path, vb_dir, cal_dir)

    print(f"\nBone Analysis {'successful' if result.success else 'failed'}")
    if result.L1:
        print(f"L1: {result.L1.vBMD_mean_mgcm3:.1f} mg/cm³ (volume: {result.L1.trabecular_volume_cm3:.2f} cm³)")
    if result.L2:
        print(f"L2: {result.L2.vBMD_mean_mgcm3:.1f} mg/cm³ (volume: {result.L2.trabecular_volume_cm3:.2f} cm³)")
    if result.L1L2_vBMD_mean_mgcm3:
        print(f"L1L2 mean: {result.L1L2_vBMD_mean_mgcm3:.1f} mg/cm³")
    print("\nQC messages:")
    for msg in result.qc_messages:
        print(f"  {msg}")
