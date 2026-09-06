#!/usr/bin/env python3
"""
bone_analysis.py - Trabecular BMD measurements from the two target vertebrae

This module handles:
1. Eroding vertebral body masks (5 mm distance transform) to isolate trabecular bone
2. Extracting HU values from trabecular regions (-50 to 400 HU)
3. Converting HU to BMD using the phantom calibration
4. Computing BMD metrics (mean, median, volume)

Calibration (changed 2026-09, review finding F02):
    If phantom_zprofile.json exists (written by 02_phantom_zprofile.py), each
    vertebra is calibrated with the rod means averaged over the slices of its own
    trabecular ROI ("co-located"). The original click-slice calibration
    (calibration_bmd.json, 9 slices around z0) is still applied and stored as
    vBMD_mean_z50_calibration_mgcm3 for the sensitivity table.

The trabecular mask that is actually measured is returned and saved (review L05),
and JSON output never contains NaN (review L06).
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

from utils.calibration import load_zprofile, calibration_for_slices, z50_calibration

logger = logging.getLogger(__name__)


# Valid HU range for trabecular bone
HU_MIN = -50
HU_MAX = 400

# Erosion distance for trabecular isolation
DEFAULT_EROSION_MM = 5.0


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
    # calibration actually used for this vertebra
    calibration_method: str = "none"
    calibration_slope: float = np.nan
    calibration_intercept: float = np.nan
    calibration_r_squared: float = np.nan
    calibration_z_min: Optional[int] = None
    calibration_z_max: Optional[int] = None
    calibration_n_slices: Optional[int] = None
    calibration_rod_mean_hu: Optional[dict] = None
    # same ROI, original click-slice calibration (sensitivity)
    vBMD_mean_z50_calibration_mgcm3: float = np.nan
    # erosion actually applied (fallbacks are recorded, not hidden)
    erosion_mm_used: float = np.nan
    roi_z_min: Optional[int] = None
    roi_z_max: Optional[int] = None


@dataclass
class BoneAnalysisResult:
    """Complete bone analysis results."""
    success: bool
    L1: Optional[VertebralBMDResult]
    L2: Optional[VertebralBMDResult]
    L1L2_vBMD_mean_mgcm3: Optional[float]
    L1L2_vBMD_weighted_mean_mgcm3: Optional[float]
    L1L2_vBMD_mean_z50_calibration_mgcm3: Optional[float]
    calibration_method: str
    calibration_slope: float          # z50 (click-slice) values, kept for backward compatibility
    calibration_intercept: float
    erosion_mm: float
    qc_messages: list = field(default_factory=list)


def erode_to_trabecular(mask: np.ndarray,
                        voxel_sizes: Tuple[float, float, float],
                        erosion_mm: float = DEFAULT_EROSION_MM) -> np.ndarray:
    """Erode a body mask by a physical distance (mm) using the distance transform."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    dist = distance_transform_edt(mask, sampling=voxel_sizes)
    return dist > erosion_mm


def _empty_result(name: str, body_volume_cm3: float = 0.0, erosion_used: float = np.nan) -> VertebralBMDResult:
    return VertebralBMDResult(
        vertebra=name, vBMD_mean_mgcm3=np.nan, vBMD_median_mgcm3=np.nan, vBMD_std_mgcm3=np.nan,
        vBMD_P10_mgcm3=np.nan, vBMD_P90_mgcm3=np.nan, trabecular_volume_cm3=0.0,
        total_body_volume_cm3=body_volume_cm3, voxel_count=0, valid_voxel_count=0,
        hu_mean=np.nan, hu_median=np.nan, erosion_mm_used=erosion_used)


def analyze_vertebra(body_nii: nib.Nifti1Image,
                     ct_nii: nib.Nifti1Image,
                     vertebra_name: str,
                     z50: dict,
                     zprofile: Optional[dict] = None,
                     erosion_mm: float = DEFAULT_EROSION_MM
                     ) -> Tuple[VertebralBMDResult, np.ndarray]:
    """
    Analyze a single vertebra. Returns (result, trabecular_mask_actually_used).

    Args:
        body_nii: body-only vertebral mask (endplates excluded)
        ct_nii: CT volume
        vertebra_name: "L1" or "L2" (pipeline names: superior / inferior target body)
        z50: click-slice calibration dict (slope, intercept, r_squared)
        zprofile: phantom_zprofile.json content; if given, co-located calibration is used
        erosion_mm: requested erosion
    """
    body_mask = np.asarray(body_nii.dataobj) > 0
    ct_data = np.asarray(ct_nii.dataobj, dtype=np.float32)
    voxel_sizes = np.array(body_nii.header.get_zooms()[:3], dtype=float)
    voxel_vol_cm3 = float(np.prod(voxel_sizes)) / 1000.0

    total_body_voxels = int(body_mask.sum())
    total_body_volume_cm3 = total_body_voxels * voxel_vol_cm3
    if total_body_voxels == 0:
        logger.warning(f"{vertebra_name}: Empty body mask")
        return _empty_result(vertebra_name), np.zeros_like(body_mask)

    # Erode; fall back explicitly and record what was used
    erosion_used = erosion_mm
    trabecular_mask = erode_to_trabecular(body_mask, tuple(voxel_sizes), erosion_mm)
    if not trabecular_mask.any():
        logger.warning(f"{vertebra_name}: No trabecular region after {erosion_mm} mm erosion, trying 2 mm")
        erosion_used = 2.0
        trabecular_mask = erode_to_trabecular(body_mask, tuple(voxel_sizes), 2.0)
        if not trabecular_mask.any():
            logger.warning(f"{vertebra_name}: Still no trabecular region, using full body (erosion 0)")
            erosion_used = 0.0
            trabecular_mask = body_mask.copy()

    hu_values = ct_data[trabecular_mask]
    valid_mask = (hu_values >= HU_MIN) & (hu_values <= HU_MAX)
    valid_hu = hu_values[valid_mask]
    zs = np.where(trabecular_mask.any(axis=(0, 1)))[0]
    roi_z = (int(zs.min()), int(zs.max()))

    if valid_hu.size == 0:
        logger.warning(f"{vertebra_name}: No valid HU values in range [{HU_MIN}, {HU_MAX}]")
        r = _empty_result(vertebra_name, total_body_volume_cm3, erosion_used)
        r.trabecular_volume_cm3 = int(trabecular_mask.sum()) * voxel_vol_cm3
        r.voxel_count = int(trabecular_mask.sum())
        r.roi_z_min, r.roi_z_max = roi_z
        return r, trabecular_mask

    # --- calibration for this ROI -------------------------------------------
    # Only voxels inside the ROI slices contribute; use the slices the valid voxels occupy.
    valid_voxel_z = np.where(trabecular_mask)[2][valid_mask]
    cal = None
    if zprofile is not None:
        try:
            cal = calibration_for_slices(zprofile, np.unique(valid_voxel_z))
        except ValueError as e:
            logger.warning(f"{vertebra_name}: co-located calibration unavailable ({e}); using click-slice calibration")
    if cal is None:
        cal = {"method": z50["method"], "slope": z50["slope"], "intercept": z50["intercept"],
               "r_squared": z50["r_squared"], "z_min": None, "z_max": None, "n_slices": None, "rod_mean_hu": None}

    bmd_values = cal["slope"] * valid_hu + cal["intercept"]
    bmd_z50 = float(z50["slope"] * np.mean(valid_hu) + z50["intercept"])

    result = VertebralBMDResult(
        vertebra=vertebra_name,
        vBMD_mean_mgcm3=float(np.mean(bmd_values)),
        vBMD_median_mgcm3=float(np.median(bmd_values)),
        vBMD_std_mgcm3=float(np.std(bmd_values)),
        vBMD_P10_mgcm3=float(np.percentile(bmd_values, 10)),
        vBMD_P90_mgcm3=float(np.percentile(bmd_values, 90)),
        trabecular_volume_cm3=valid_hu.size * voxel_vol_cm3,
        total_body_volume_cm3=total_body_volume_cm3,
        voxel_count=int(trabecular_mask.sum()),
        valid_voxel_count=int(valid_hu.size),
        hu_mean=float(np.mean(valid_hu)),
        hu_median=float(np.median(valid_hu)),
        calibration_method=cal["method"],
        calibration_slope=cal["slope"],
        calibration_intercept=cal["intercept"],
        calibration_r_squared=cal["r_squared"],
        calibration_z_min=cal.get("z_min"),
        calibration_z_max=cal.get("z_max"),
        calibration_n_slices=cal.get("n_slices"),
        calibration_rod_mean_hu=cal.get("rod_mean_hu"),
        vBMD_mean_z50_calibration_mgcm3=bmd_z50,
        erosion_mm_used=erosion_used,
        roi_z_min=roi_z[0], roi_z_max=roi_z[1],
    )
    logger.info(f"{vertebra_name}: vBMD mean={result.vBMD_mean_mgcm3:.1f} mg/cm³ "
                f"(click-slice calibration would give {bmd_z50:.1f}), "
                f"volume={result.trabecular_volume_cm3:.2f} cm³, slope={cal['slope']:.4f} "
                f"[{cal['method']}]")
    return result, trabecular_mask


def analyze_bone(ct_path: Path,
                 vertebral_bodies_dir: Path,
                 calibration_dir: Path,
                 erosion_mm: float = DEFAULT_EROSION_MM,
                 save_trabecular_masks: bool = True) -> BoneAnalysisResult:
    """
    Run complete bone analysis for the two target vertebrae.

    Args:
        ct_path: Path to CT NIfTI file
        vertebral_bodies_dir: Directory containing L1_body.nii.gz and L2_body.nii.gz
        calibration_dir: Directory containing calibration_bmd.json (and phantom_zprofile.json)
        erosion_mm: Erosion distance for trabecular isolation
    """
    qc_messages = []
    cal_path = Path(calibration_dir) / "calibration_bmd.json"
    if not cal_path.exists():
        logger.error(f"Calibration file not found: {cal_path}")
        return BoneAnalysisResult(False, None, None, None, None, None, "none", 0.0, 0.0,
                                  erosion_mm, ["Calibration file not found"])
    z50 = z50_calibration(calibration_dir)
    zprofile = load_zprofile(calibration_dir)
    if zprofile is None:
        qc_messages.append("WARNING: phantom_zprofile.json missing; click-slice calibration used for both vertebrae")
        method = z50["method"]
    else:
        method = "co-located (per-vertebra, tracked rods over ROI slices)"
        if zprofile.get("needs_review"):
            qc_messages.append("WARNING: phantom tracking flagged for review: "
                               + ", ".join(k for k, v in zprofile["flags"].items() if v))
    logger.info(f"Click-slice calibration: slope={z50['slope']:.4f}, intercept={z50['intercept']:.2f}, "
                f"R²={z50['r_squared']:.4f}; method in use: {method}")

    ct_nii = nib.load(ct_path)
    trabecular_dir = Path(vertebral_bodies_dir) / "trabecular_masks"
    if save_trabecular_masks:
        trabecular_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Optional[VertebralBMDResult]] = {"L1": None, "L2": None}
    for name in ["L1", "L2"]:
        body_path = Path(vertebral_bodies_dir) / f"{name}_body.nii.gz"
        if not body_path.exists():
            qc_messages.append(f"ERROR: {name} body mask not found")
            continue
        body_nii = nib.load(body_path)
        res, used_mask = analyze_vertebra(body_nii, ct_nii, name, z50, zprofile, erosion_mm)
        results[name] = res
        if save_trabecular_masks:
            nib.save(nib.Nifti1Image(used_mask.astype(np.uint8), body_nii.affine, body_nii.header),
                     trabecular_dir / f"{name}_trabecular.nii.gz")
        if np.isnan(res.vBMD_mean_mgcm3):
            qc_messages.append(f"WARNING: {name} BMD could not be calculated")
        elif res.vBMD_mean_mgcm3 < 50 or res.vBMD_mean_mgcm3 > 250:
            qc_messages.append(f"WARNING: {name} BMD = {res.vBMD_mean_mgcm3:.1f} mg/cm³ outside typical range (50-250)")
        if res.erosion_mm_used != erosion_mm:
            qc_messages.append(f"WARNING: {name} erosion fell back to {res.erosion_mm_used} mm")
        if res.calibration_r_squared < 0.99:
            qc_messages.append(f"WARNING: {name} calibration R² = {res.calibration_r_squared:.4f} < 0.99")

    l1, l2 = results["L1"], results["L2"]
    l1l2_mean = l1l2_weighted = l1l2_z50 = None
    if l1 and l2 and not np.isnan(l1.vBMD_mean_mgcm3) and not np.isnan(l2.vBMD_mean_mgcm3):
        l1l2_mean = (l1.vBMD_mean_mgcm3 + l2.vBMD_mean_mgcm3) / 2
        l1l2_z50 = (l1.vBMD_mean_z50_calibration_mgcm3 + l2.vBMD_mean_z50_calibration_mgcm3) / 2
        total_vol = l1.trabecular_volume_cm3 + l2.trabecular_volume_cm3
        if total_vol > 0:
            l1l2_weighted = (l1.vBMD_mean_mgcm3 * l1.trabecular_volume_cm3 +
                             l2.vBMD_mean_mgcm3 * l2.trabecular_volume_cm3) / total_vol
        logger.info(f"L1L2 combined: mean={l1l2_mean:.1f} mg/cm³ (click-slice calibration {l1l2_z50:.1f})")

    success = l1l2_mean is not None
    return BoneAnalysisResult(
        success=success, L1=l1, L2=l2,
        L1L2_vBMD_mean_mgcm3=l1l2_mean,
        L1L2_vBMD_weighted_mean_mgcm3=l1l2_weighted,
        L1L2_vBMD_mean_z50_calibration_mgcm3=l1l2_z50,
        calibration_method=method,
        calibration_slope=z50["slope"], calibration_intercept=z50["intercept"],
        erosion_mm=erosion_mm, qc_messages=qc_messages)


def _json_safe(obj):
    """numpy -> python, non-finite floats -> None (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def save_bone_results(result: BoneAnalysisResult, output_path: Path):
    """Save bone analysis results to JSON (strict: no NaN)."""
    out = {
        "success": result.success,
        "L1L2_vBMD_mean_mgcm3": result.L1L2_vBMD_mean_mgcm3,
        "L1L2_vBMD_weighted_mean_mgcm3": result.L1L2_vBMD_weighted_mean_mgcm3,
        "L1L2_vBMD_mean_z50_calibration_mgcm3": result.L1L2_vBMD_mean_z50_calibration_mgcm3,
        "calibration_method": result.calibration_method,
        "calibration": {"slope": result.calibration_slope, "intercept": result.calibration_intercept,
                        "note": "click-slice (z0) calibration kept for reference; per-vertebra "
                                "calibration is inside L1/L2 when calibration_method is co-located"},
        "erosion_mm": result.erosion_mm,
        "hu_range": [HU_MIN, HU_MAX],
        "qc_messages": result.qc_messages,
    }
    if result.L1:
        out["L1"] = asdict(result.L1)
    if result.L2:
        out["L2"] = asdict(result.L2)
    with open(output_path, "w") as f:
        json.dump(_json_safe(out), f, indent=2, allow_nan=False)
    logger.info(f"Bone results saved to {output_path}")


def get_trabecular_mask(body_nii: nib.Nifti1Image,
                        erosion_mm: float = DEFAULT_EROSION_MM) -> nib.Nifti1Image:
    """Trabecular mask (requested erosion, no fallback) for QC visualisation."""
    body_mask = np.asarray(body_nii.dataobj) > 0
    voxel_sizes = body_nii.header.get_zooms()[:3]
    trabecular = erode_to_trabecular(body_mask, voxel_sizes, erosion_mm)
    return nib.Nifti1Image(trabecular.astype(np.uint8), body_nii.affine, body_nii.header)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 4:
        print("Usage: python 03_bone_analysis.py <ct_path> <vertebral_bodies_dir> <calibration_dir>")
        sys.exit(1)
    result = analyze_bone(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(f"\nBone Analysis {'successful' if result.success else 'failed'} [{result.calibration_method}]")
    for name in ["L1", "L2"]:
        r = getattr(result, name)
        if r:
            print(f"{name}: {r.vBMD_mean_mgcm3:.1f} mg/cm³ (click-slice cal: {r.vBMD_mean_z50_calibration_mgcm3:.1f}), "
                  f"volume {r.trabecular_volume_cm3:.2f} cm³, slope {r.calibration_slope:.4f}")
    if result.L1L2_vBMD_mean_mgcm3:
        print(f"L1L2 mean: {result.L1L2_vBMD_mean_mgcm3:.1f} mg/cm³")
    for msg in result.qc_messages:
        print(f"  {msg}")
