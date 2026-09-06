#!/usr/bin/env python3
"""
muscle_analysis.py - Skeletal muscle composition analysis

This module handles:
1. Classifying voxels within the muscle compartment envelope by HU
2. Computing SMD (skeletal muscle density)
3. Measuring IMAT (intermuscular adipose tissue)
4. Calculating myosteatosis metrics
5. Computing cross-sectional areas

Changes 2026-09 (adversarial review):
- HU classes are continuous half-open intervals (review F03). The drift offset is
  fractional, so the old closed integer intervals (-29..29, 30..150) left voxels
  in (29, 30) and (-30, -29) unclassified.
      IMAT            -190 <= HU < -30
      low-density      -30 <= HU <  30
      normal            30 <= HU <= 150
      muscle (all)     -30 <= HU <= 150
- Drift offset is the phantom base-material mean over the slices of the muscle
  slab (co-located, from phantom_zprofile.json) instead of the 9 click slices
  (review F02). The click-slice value is kept in the output for reference.
- The L/R symmetry index is computed inside the envelope (review F08).
- JSON output never contains NaN (review L06).
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict

import numpy as np
import nibabel as nib

from utils.calibration import load_zprofile, base_offset_for_slices

logger = logging.getLogger(__name__)


# HU class boundaries (applied after drift correction). Half-open at the
# internal boundaries so that every HU value in [-190, 150] belongs to exactly
# one class; see module docstring.
HU_IMAT_LOW = -190.0
HU_IMAT_HIGH = -30.0      # exclusive
HU_LOW_HIGH = 30.0        # exclusive
HU_MUSCLE_HIGH = 150.0    # inclusive

CLASSIFICATION_CONVENTION = ("IMAT [-190,-30) HU; low-density muscle [-30,30) HU; "
                             "normal muscle [30,150] HU; muscle_all [-30,150] HU; "
                             "after adding the drift offset")

# Kept for documentation / backward-compatible imports
THRESHOLDS = {
    "imat": (HU_IMAT_LOW, HU_IMAT_HIGH),
    "muscle_low": (HU_IMAT_HIGH, HU_LOW_HIGH),
    "muscle_normal": (HU_LOW_HIGH, HU_MUSCLE_HIGH),
    "muscle_all": (HU_IMAT_HIGH, HU_MUSCLE_HIGH),
}


@dataclass
class MuscleAnalysisResult:
    """Results from muscle composition analysis."""
    success: bool

    # Volume metrics (cm³)
    compartment_volume_cm3: float
    muscle_tissue_volume_cm3: float
    muscle_normal_volume_cm3: float
    muscle_low_density_volume_cm3: float
    imat_volume_cm3: float

    # Percentage metrics
    muscle_low_density_percent: float
    imat_percent: float

    # Density metrics (HU)
    muscle_SMD_mean_hu: float
    muscle_SMD_median_hu: float
    muscle_SMD_std_hu: float
    muscle_SMD_P10_hu: float
    muscle_SMD_P90_hu: float

    # Cross-sectional area (cm²)
    muscle_CSA_mean_cm2: float
    muscle_CSA_max_cm2: float

    # Symmetry QC (within envelope)
    muscle_LR_symmetry_index: float

    # Drift correction actually used, and the click-slice value for reference
    drift_correction_hu: float
    drift_correction_method: str
    drift_correction_hu_z50: float
    drift_slices: Optional[list]

    n_slices: int
    envelope_z_min: Optional[int]
    envelope_z_max: Optional[int]
    classification_convention: str
    qc_messages: list


def classify_voxels(ct_data: np.ndarray,
                    envelope_mask: np.ndarray,
                    drift_offset: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Classify voxels within the muscle compartment by tissue type using the
    continuous half-open convention in the module docstring.

    Args:
        ct_data: CT volume (raw HU)
        envelope_mask: Muscle compartment envelope mask
        drift_offset: HU ADDED to the data before classification
    """
    corrected = ct_data + drift_offset
    inside = envelope_mask.astype(bool)
    imat = inside & (corrected >= HU_IMAT_LOW) & (corrected < HU_IMAT_HIGH)
    low = inside & (corrected >= HU_IMAT_HIGH) & (corrected < HU_LOW_HIGH)
    normal = inside & (corrected >= HU_LOW_HIGH) & (corrected <= HU_MUSCLE_HIGH)
    return {"imat": imat, "muscle_low": low, "muscle_normal": normal, "muscle_all": low | normal}


def compute_symmetry_index(left_mask: np.ndarray,
                           right_mask: np.ndarray,
                           envelope_mask: np.ndarray) -> float:
    """Ratio of smaller to larger side volume, restricted to the envelope (1.0 = symmetric)."""
    env = envelope_mask.astype(bool)
    left_vol = np.count_nonzero(left_mask & env)
    right_vol = np.count_nonzero(right_mask & env)
    if left_vol == 0 or right_vol == 0:
        return 0.0
    return min(left_vol, right_vol) / max(left_vol, right_vol)


def compute_csa(mask: np.ndarray, voxel_sizes: Tuple[float, float, float]) -> Tuple[float, float]:
    """Mean and max cross-sectional area (cm²) over slices that contain the mask."""
    voxel_area_cm2 = voxel_sizes[0] * voxel_sizes[1] / 100.0
    per_slice = mask.sum(axis=(0, 1))
    per_slice = per_slice[per_slice > 0] * voxel_area_cm2
    if per_slice.size == 0:
        return 0.0, 0.0
    return float(np.mean(per_slice)), float(np.max(per_slice))


def resolve_drift_offset(calibration_dir: Path, envelope_mask: Optional[np.ndarray]) -> dict:
    """
    Decide the drift offset for a session.

    Returns dict with offset_hu (used), method, offset_hu_z50, slices.
    Co-located (phantom base over the envelope slices) when phantom_zprofile.json
    exists and the envelope is non-empty; otherwise the click-slice value from
    calibration_hu_stability.json.
    """
    z50 = None
    stability_path = Path(calibration_dir) / "calibration_hu_stability.json"
    if stability_path.exists():
        with open(stability_path) as f:
            z50 = float(json.load(f).get("drift_correction", {}).get("offset_hu", 0.0))

    zprofile = load_zprofile(calibration_dir)
    if zprofile is not None and envelope_mask is not None and envelope_mask.any():
        zs = np.where(envelope_mask.any(axis=(0, 1)))[0]
        try:
            info = base_offset_for_slices(zprofile, zs)
            return {"offset_hu": info["offset_hu"], "method": info["method"],
                    "offset_hu_z50": z50 if z50 is not None else np.nan,
                    "slices": [info["z_min"], info["z_max"]], "n_slices": info["n_slices"]}
        except ValueError as e:
            logger.warning(f"co-located drift offset unavailable ({e}); using click-slice value")
    if z50 is None:
        return {"offset_hu": 0.0, "method": "none (no calibration available)", "offset_hu_z50": np.nan,
                "slices": None, "n_slices": 0}
    return {"offset_hu": z50, "method": "click slice (calibration_hu_stability.json)",
            "offset_hu_z50": z50, "slices": None, "n_slices": 9}


def analyze_muscle(ct_path: Path,
                   envelope_path: Path,
                   segmentations_dir: Path,
                   calibration_dir: Path) -> MuscleAnalysisResult:
    """
    Run complete muscle composition analysis.

    Args:
        ct_path: Path to CT NIfTI file
        envelope_path: Path to muscle compartment envelope
        segmentations_dir: Directory with erector spinae masks (for symmetry)
        calibration_dir: Directory containing calibration_hu_stability.json / phantom_zprofile.json
    """
    qc_messages = []
    ct_nii = nib.load(ct_path)
    ct_data = np.asarray(ct_nii.dataobj, dtype=np.float32)
    voxel_sizes = ct_nii.header.get_zooms()[:3]
    envelope_mask = np.asarray(nib.load(envelope_path).dataobj) > 0

    drift = resolve_drift_offset(calibration_dir, envelope_mask)
    drift_offset = float(drift["offset_hu"])
    logger.info(f"Drift correction: {drift_offset:+.2f} HU [{drift['method']}] "
                f"(click-slice value {drift['offset_hu_z50']:+.2f} HU)")
    if drift["method"].startswith("none"):
        qc_messages.append("WARNING: No drift correction available")

    if not envelope_mask.any():
        logger.error("Empty muscle envelope")
        return MuscleAnalysisResult(
            success=False, compartment_volume_cm3=0.0, muscle_tissue_volume_cm3=0.0,
            muscle_normal_volume_cm3=0.0, muscle_low_density_volume_cm3=0.0, imat_volume_cm3=0.0,
            muscle_low_density_percent=0.0, imat_percent=0.0,
            muscle_SMD_mean_hu=np.nan, muscle_SMD_median_hu=np.nan, muscle_SMD_std_hu=np.nan,
            muscle_SMD_P10_hu=np.nan, muscle_SMD_P90_hu=np.nan,
            muscle_CSA_mean_cm2=0.0, muscle_CSA_max_cm2=0.0, muscle_LR_symmetry_index=0.0,
            drift_correction_hu=drift_offset, drift_correction_method=drift["method"],
            drift_correction_hu_z50=drift["offset_hu_z50"], drift_slices=drift["slices"],
            n_slices=0, envelope_z_min=None, envelope_z_max=None,
            classification_convention=CLASSIFICATION_CONVENTION,
            qc_messages=["ERROR: Empty muscle envelope"])

    voxel_vol_cm3 = float(np.prod(voxel_sizes)) / 1000.0
    compartment_volume_cm3 = int(envelope_mask.sum()) * voxel_vol_cm3
    env_z = np.where(envelope_mask.any(axis=(0, 1)))[0]

    tissue_masks = classify_voxels(ct_data, envelope_mask, drift_offset)
    muscle_all_vol = int(tissue_masks["muscle_all"].sum()) * voxel_vol_cm3
    muscle_normal_vol = int(tissue_masks["muscle_normal"].sum()) * voxel_vol_cm3
    muscle_low_vol = int(tissue_masks["muscle_low"].sum()) * voxel_vol_cm3
    imat_vol = int(tissue_masks["imat"].sum()) * voxel_vol_cm3

    muscle_low_pct = (muscle_low_vol / muscle_all_vol * 100) if muscle_all_vol > 0 else 0.0
    imat_pct = (imat_vol / compartment_volume_cm3 * 100) if compartment_volume_cm3 > 0 else 0.0

    muscle_hu = (ct_data + drift_offset)[tissue_masks["muscle_all"]]
    if muscle_hu.size > 0:
        smd_mean, smd_median, smd_std = float(np.mean(muscle_hu)), float(np.median(muscle_hu)), float(np.std(muscle_hu))
        smd_p10, smd_p90 = float(np.percentile(muscle_hu, 10)), float(np.percentile(muscle_hu, 90))
    else:
        smd_mean = smd_median = smd_std = smd_p10 = smd_p90 = np.nan
        qc_messages.append("WARNING: No muscle voxels found")

    csa_mean, csa_max = compute_csa(tissue_masks["muscle_all"], voxel_sizes)
    n_slices = int(np.count_nonzero(tissue_masks["muscle_all"].any(axis=(0, 1))))

    # Symmetry inside the envelope
    symmetry_index = 0.0
    pair = None
    for left_name, right_name in [("erector_spinae_left.nii.gz", "erector_spinae_right.nii.gz"),
                                  ("autochthon_left.nii.gz", "autochthon_right.nii.gz")]:
        l_path, r_path = Path(segmentations_dir) / left_name, Path(segmentations_dir) / right_name
        if l_path.exists() and r_path.exists():
            pair = (l_path, r_path)
            break
    if pair:
        left_mask = np.asarray(nib.load(pair[0]).dataobj) > 0
        right_mask = np.asarray(nib.load(pair[1]).dataobj) > 0
        symmetry_index = compute_symmetry_index(left_mask, right_mask, envelope_mask)
        if symmetry_index < 0.7:
            qc_messages.append(f"WARNING: Low L/R symmetry = {symmetry_index:.2f}")
    else:
        qc_messages.append("WARNING: Could not compute symmetry (missing L/R masks)")

    if imat_pct == 0:
        qc_messages.append("WARNING: IMAT = 0% (may indicate envelope problem)")
    if muscle_low_pct > 50:
        qc_messages.append(f"WARNING: High myosteatosis = {muscle_low_pct:.1f}%")

    logger.info(f"Muscle analysis: SMD={smd_mean:.1f} HU, low density={muscle_low_pct:.1f}%, "
                f"IMAT={imat_pct:.1f}%, volume={muscle_all_vol:.1f} cm³, CSA={csa_mean:.1f} cm²")

    return MuscleAnalysisResult(
        success=True,
        compartment_volume_cm3=compartment_volume_cm3,
        muscle_tissue_volume_cm3=muscle_all_vol,
        muscle_normal_volume_cm3=muscle_normal_vol,
        muscle_low_density_volume_cm3=muscle_low_vol,
        imat_volume_cm3=imat_vol,
        muscle_low_density_percent=muscle_low_pct,
        imat_percent=imat_pct,
        muscle_SMD_mean_hu=smd_mean, muscle_SMD_median_hu=smd_median, muscle_SMD_std_hu=smd_std,
        muscle_SMD_P10_hu=smd_p10, muscle_SMD_P90_hu=smd_p90,
        muscle_CSA_mean_cm2=csa_mean, muscle_CSA_max_cm2=csa_max,
        muscle_LR_symmetry_index=symmetry_index,
        drift_correction_hu=drift_offset,
        drift_correction_method=drift["method"],
        drift_correction_hu_z50=drift["offset_hu_z50"],
        drift_slices=drift["slices"],
        n_slices=n_slices,
        envelope_z_min=int(env_z.min()), envelope_z_max=int(env_z.max()),
        classification_convention=CLASSIFICATION_CONVENTION,
        qc_messages=qc_messages)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_muscle_results(result: MuscleAnalysisResult, output_path: Path):
    with open(output_path, "w") as f:
        json.dump(_json_safe(asdict(result)), f, indent=2, allow_nan=False)
    logger.info(f"Muscle results saved to {output_path}")


def _load_for_masks(ct_path: Path, envelope_path: Path, calibration_dir: Path):
    ct_nii = nib.load(ct_path)
    ct_data = np.asarray(ct_nii.dataobj, dtype=np.float32)
    envelope_mask = np.asarray(nib.load(envelope_path).dataobj) > 0
    drift = resolve_drift_offset(calibration_dir, envelope_mask)
    return ct_nii, ct_data, envelope_mask, float(drift["offset_hu"])


def get_tissue_classification_mask(ct_path: Path,
                                   envelope_path: Path,
                                   calibration_dir: Path) -> nib.Nifti1Image:
    """Labelled mask for QC: 1 = IMAT, 2 = low-density muscle, 3 = normal muscle."""
    ct_nii, ct_data, envelope_mask, drift_offset = _load_for_masks(ct_path, envelope_path, calibration_dir)
    tissue_masks = classify_voxels(ct_data, envelope_mask, drift_offset)
    labels = np.zeros(ct_data.shape, dtype=np.uint8)
    labels[tissue_masks["imat"]] = 1
    labels[tissue_masks["muscle_low"]] = 2
    labels[tissue_masks["muscle_normal"]] = 3
    return nib.Nifti1Image(labels, ct_nii.affine, ct_nii.header)


def save_tissue_masks(ct_path: Path,
                      envelope_path: Path,
                      calibration_dir: Path,
                      output_dir: Path) -> Dict[str, Path]:
    """
    Save individual tissue classification masks as NIfTI files:
    muscle_envelope, muscle_imat, muscle_low_density, muscle_normal, muscle_all,
    muscle_classification (1=IMAT, 2=low, 3=normal).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ct_nii, ct_data, envelope_mask, drift_offset = _load_for_masks(ct_path, envelope_path, calibration_dir)
    tissue_masks = classify_voxels(ct_data, envelope_mask, drift_offset)

    def save(mask, name):
        p = output_dir / name
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), ct_nii.affine, ct_nii.header), p)
        return p

    saved = {
        "envelope": save(envelope_mask, "muscle_envelope.nii.gz"),
        "imat": save(tissue_masks["imat"], "muscle_imat.nii.gz"),
        "muscle_low": save(tissue_masks["muscle_low"], "muscle_low_density.nii.gz"),
        "muscle_normal": save(tissue_masks["muscle_normal"], "muscle_normal.nii.gz"),
        "muscle_all": save(tissue_masks["muscle_all"], "muscle_all.nii.gz"),
    }
    labels = np.zeros(ct_data.shape, dtype=np.uint8)
    labels[tissue_masks["imat"]] = 1
    labels[tissue_masks["muscle_low"]] = 2
    labels[tissue_masks["muscle_normal"]] = 3
    saved["classification"] = save(labels, "muscle_classification.nii.gz")
    logger.info(f"Saved {len(saved)} muscle tissue masks to {output_dir}")
    return saved


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 5:
        print("Usage: python 04_muscle_analysis.py <ct_path> <envelope_path> <seg_dir> <cal_dir>")
        sys.exit(1)
    result = analyze_muscle(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    print(f"\nMuscle Analysis {'successful' if result.success else 'failed'}")
    print(f"SMD: {result.muscle_SMD_mean_hu:.1f} HU, low density: {result.muscle_low_density_percent:.1f}%, "
          f"IMAT: {result.imat_percent:.1f}%, volume: {result.muscle_tissue_volume_cm3:.1f} cm³, "
          f"CSA: {result.muscle_CSA_mean_cm2:.1f} cm², drift {result.drift_correction_hu:+.2f} HU "
          f"[{result.drift_correction_method}]")
    for msg in result.qc_messages:
        print(f"  {msg}")
