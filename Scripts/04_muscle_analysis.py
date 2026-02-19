#!/usr/bin/env python3
"""
muscle_analysis.py - Skeletal muscle composition analysis

This module handles:
1. Classifying voxels within muscle compartment envelope by HU
2. Computing SMD (skeletal muscle density)
3. Measuring IMAT (intermuscular adipose tissue)
4. Calculating myosteatosis metrics
5. Computing cross-sectional areas
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict

import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)


# HU thresholds for tissue classification (standard literature values)
# Note: These are applied after drift correction
THRESHOLDS = {
    "imat": (-190, -30),           # Intermuscular adipose tissue
    "muscle_low": (-29, 29),       # Low density (myosteatotic) muscle
    "muscle_normal": (30, 150),    # Normal density muscle
    "muscle_all": (-29, 150),      # All muscle tissue
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

    # Symmetry QC
    muscle_LR_symmetry_index: float

    # Additional info
    drift_correction_hu: float
    n_slices: int
    qc_messages: list


def classify_voxels(ct_data: np.ndarray,
                    envelope_mask: np.ndarray,
                    drift_offset: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Classify voxels within muscle compartment by tissue type.

    Args:
        ct_data: CT volume
        envelope_mask: Muscle compartment envelope mask
        drift_offset: HU drift correction (ADDED to data)
                     This is the value from calibration_hu_stability.json offset_hu,
                     which represents what to ADD to correct scanner drift.

    Returns:
        Dictionary of tissue masks
    """
    # Apply drift correction (add offset to correct scanner drift)
    # offset_hu is computed as -drift_hu, i.e., if base reads -10 HU,
    # offset_hu = +10, and we ADD it to bring measurements back to nominal
    corrected = ct_data + drift_offset

    # Initialize output masks
    masks = {}

    # Get envelope voxels
    in_envelope = envelope_mask.astype(bool)

    # Classify by HU thresholds
    for tissue, (low, high) in THRESHOLDS.items():
        mask = in_envelope & (corrected >= low) & (corrected <= high)
        masks[tissue] = mask

    return masks


def compute_symmetry_index(left_mask: np.ndarray,
                           right_mask: np.ndarray,
                           envelope_mask: np.ndarray) -> float:
    """
    Compute left-right symmetry index for QC.

    Args:
        left_mask: Left erector spinae mask
        right_mask: Right erector spinae mask
        envelope_mask: Combined envelope mask

    Returns:
        Symmetry index (ratio of smaller/larger side volume)
    """
    left_vol = left_mask.sum()
    right_vol = right_mask.sum()

    if left_vol == 0 or right_vol == 0:
        return 0.0

    # Ratio of smaller to larger (1.0 = perfect symmetry)
    return min(left_vol, right_vol) / max(left_vol, right_vol)


def compute_csa(mask: np.ndarray, voxel_sizes: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Compute cross-sectional area statistics.

    Args:
        mask: 3D binary mask
        voxel_sizes: Voxel dimensions (x, y, z) in mm

    Returns:
        Tuple of (mean_CSA_cm2, max_CSA_cm2)
    """
    voxel_area_mm2 = voxel_sizes[0] * voxel_sizes[1]
    voxel_area_cm2 = voxel_area_mm2 / 100.0

    # Calculate area for each slice
    slice_areas = []
    for z in range(mask.shape[2]):
        slice_mask = mask[:, :, z]
        if slice_mask.any():
            area = slice_mask.sum() * voxel_area_cm2
            slice_areas.append(area)

    if not slice_areas:
        return 0.0, 0.0

    return np.mean(slice_areas), np.max(slice_areas)


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
        calibration_dir: Directory containing calibration_hu_stability.json

    Returns:
        MuscleAnalysisResult with all metrics
    """
    qc_messages = []

    # Load drift correction
    drift_offset = 0.0
    stability_path = calibration_dir / "calibration_hu_stability.json"
    if stability_path.exists():
        with open(stability_path) as f:
            stability = json.load(f)
        drift_offset = stability.get("drift_correction", {}).get("offset_hu", 0.0)
        logger.info(f"Drift correction: {drift_offset:.1f} HU")
    else:
        qc_messages.append("WARNING: No drift correction available")

    # Load CT and envelope
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()
    voxel_sizes = ct_nii.header.get_zooms()[:3]

    envelope_nii = nib.load(envelope_path)
    envelope_mask = envelope_nii.get_fdata().astype(bool)

    if not envelope_mask.any():
        logger.error("Empty muscle envelope")
        return MuscleAnalysisResult(
            success=False,
            compartment_volume_cm3=0.0,
            muscle_tissue_volume_cm3=0.0,
            muscle_normal_volume_cm3=0.0,
            muscle_low_density_volume_cm3=0.0,
            imat_volume_cm3=0.0,
            muscle_low_density_percent=0.0,
            imat_percent=0.0,
            muscle_SMD_mean_hu=np.nan,
            muscle_SMD_median_hu=np.nan,
            muscle_SMD_std_hu=np.nan,
            muscle_SMD_P10_hu=np.nan,
            muscle_SMD_P90_hu=np.nan,
            muscle_CSA_mean_cm2=0.0,
            muscle_CSA_max_cm2=0.0,
            muscle_LR_symmetry_index=0.0,
            drift_correction_hu=drift_offset,
            n_slices=0,
            qc_messages=["ERROR: Empty muscle envelope"]
        )

    # Calculate voxel volume
    voxel_vol_mm3 = np.prod(voxel_sizes)
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

    # Compartment volume
    compartment_voxels = envelope_mask.sum()
    compartment_volume_cm3 = compartment_voxels * voxel_vol_cm3

    # Classify voxels
    tissue_masks = classify_voxels(ct_data, envelope_mask, drift_offset)

    # Calculate volumes
    muscle_all_vol = tissue_masks["muscle_all"].sum() * voxel_vol_cm3
    muscle_normal_vol = tissue_masks["muscle_normal"].sum() * voxel_vol_cm3
    muscle_low_vol = tissue_masks["muscle_low"].sum() * voxel_vol_cm3
    imat_vol = tissue_masks["imat"].sum() * voxel_vol_cm3

    # Calculate percentages
    muscle_low_pct = 0.0
    if muscle_all_vol > 0:
        muscle_low_pct = (muscle_low_vol / muscle_all_vol) * 100

    imat_pct = 0.0
    if compartment_volume_cm3 > 0:
        imat_pct = (imat_vol / compartment_volume_cm3) * 100

    # Extract muscle HU values for SMD calculation
    corrected_ct = ct_data + drift_offset
    muscle_hu = corrected_ct[tissue_masks["muscle_all"]]

    if len(muscle_hu) > 0:
        smd_mean = float(np.mean(muscle_hu))
        smd_median = float(np.median(muscle_hu))
        smd_std = float(np.std(muscle_hu))
        smd_p10 = float(np.percentile(muscle_hu, 10))
        smd_p90 = float(np.percentile(muscle_hu, 90))
    else:
        smd_mean = smd_median = smd_std = smd_p10 = smd_p90 = np.nan
        qc_messages.append("WARNING: No muscle voxels found")

    # Cross-sectional area
    csa_mean, csa_max = compute_csa(tissue_masks["muscle_all"], voxel_sizes)

    # Count slices
    z_with_muscle = np.where(tissue_masks["muscle_all"].any(axis=(0, 1)))[0]
    n_slices = len(z_with_muscle)

    # Symmetry index - try multiple naming conventions
    symmetry_index = 0.0
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

    if left_path is not None and right_path is not None:
        left_mask = nib.load(left_path).get_fdata().astype(bool)
        right_mask = nib.load(right_path).get_fdata().astype(bool)
        symmetry_index = compute_symmetry_index(left_mask, right_mask, envelope_mask)

        if symmetry_index < 0.7:
            qc_messages.append(f"WARNING: Low L/R symmetry = {symmetry_index:.2f}")
        elif symmetry_index > 1.3:
            qc_messages.append(f"WARNING: High L/R symmetry = {symmetry_index:.2f}")
    else:
        qc_messages.append("WARNING: Could not compute symmetry (missing L/R masks)")

    # Additional QC checks
    if imat_pct == 0:
        qc_messages.append("WARNING: IMAT = 0% (may indicate envelope problem)")

    if muscle_low_pct > 50:
        qc_messages.append(f"WARNING: High myosteatosis = {muscle_low_pct:.1f}%")

    logger.info(f"Muscle analysis: SMD={smd_mean:.1f} HU, low density={muscle_low_pct:.1f}%, "
                f"IMAT={imat_pct:.1f}%, volume={muscle_all_vol:.1f} cm³")

    return MuscleAnalysisResult(
        success=True,
        compartment_volume_cm3=compartment_volume_cm3,
        muscle_tissue_volume_cm3=muscle_all_vol,
        muscle_normal_volume_cm3=muscle_normal_vol,
        muscle_low_density_volume_cm3=muscle_low_vol,
        imat_volume_cm3=imat_vol,
        muscle_low_density_percent=muscle_low_pct,
        imat_percent=imat_pct,
        muscle_SMD_mean_hu=smd_mean,
        muscle_SMD_median_hu=smd_median,
        muscle_SMD_std_hu=smd_std,
        muscle_SMD_P10_hu=smd_p10,
        muscle_SMD_P90_hu=smd_p90,
        muscle_CSA_mean_cm2=csa_mean,
        muscle_CSA_max_cm2=csa_max,
        muscle_LR_symmetry_index=symmetry_index,
        drift_correction_hu=drift_offset,
        n_slices=n_slices,
        qc_messages=qc_messages
    )


def save_muscle_results(result: MuscleAnalysisResult, output_path: Path):
    """Save muscle analysis results to JSON."""
    with open(output_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    logger.info(f"Muscle results saved to {output_path}")


def get_tissue_classification_mask(ct_path: Path,
                                   envelope_path: Path,
                                   calibration_dir: Path) -> nib.Nifti1Image:
    """
    Generate labeled mask for QC visualization.

    Labels:
    - 0: Outside envelope
    - 1: IMAT
    - 2: Low density muscle
    - 3: Normal density muscle

    Args:
        ct_path: Path to CT NIfTI
        envelope_path: Path to envelope mask
        calibration_dir: Directory with calibration

    Returns:
        NIfTI image with tissue labels
    """
    # Load drift correction
    drift_offset = 0.0
    stability_path = calibration_dir / "calibration_hu_stability.json"
    if stability_path.exists():
        with open(stability_path) as f:
            stability = json.load(f)
        drift_offset = stability.get("drift_correction", {}).get("offset_hu", 0.0)

    # Load data
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()

    envelope_nii = nib.load(envelope_path)
    envelope_mask = envelope_nii.get_fdata().astype(bool)

    # Classify
    tissue_masks = classify_voxels(ct_data, envelope_mask, drift_offset)

    # Create labeled output
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
    Save individual tissue classification masks as NIfTI files.

    Creates:
    - muscle_envelope.nii.gz: Full muscle compartment envelope
    - muscle_imat.nii.gz: IMAT mask (-190 to -30 HU)
    - muscle_low_density.nii.gz: Low density muscle mask (-29 to +29 HU)
    - muscle_normal.nii.gz: Normal density muscle mask (+30 to +150 HU)
    - muscle_all.nii.gz: All muscle tissue mask (-29 to +150 HU)
    - muscle_classification.nii.gz: Combined labeled mask (1=IMAT, 2=low, 3=normal)

    Args:
        ct_path: Path to CT NIfTI file
        envelope_path: Path to muscle compartment envelope
        calibration_dir: Directory containing calibration_hu_stability.json
        output_dir: Directory to save masks

    Returns:
        Dictionary mapping mask names to saved paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load drift correction
    drift_offset = 0.0
    stability_path = calibration_dir / "calibration_hu_stability.json"
    if stability_path.exists():
        with open(stability_path) as f:
            stability = json.load(f)
        drift_offset = stability.get("drift_correction", {}).get("offset_hu", 0.0)

    # Load data
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()

    envelope_nii = nib.load(envelope_path)
    envelope_mask = envelope_nii.get_fdata().astype(bool)

    # Classify voxels
    tissue_masks = classify_voxels(ct_data, envelope_mask, drift_offset)

    saved_paths = {}

    # Save envelope
    envelope_out = output_dir / "muscle_envelope.nii.gz"
    nib.save(nib.Nifti1Image(envelope_mask.astype(np.uint8), ct_nii.affine, ct_nii.header),
             envelope_out)
    saved_paths["envelope"] = envelope_out

    # Save IMAT mask
    imat_out = output_dir / "muscle_imat.nii.gz"
    nib.save(nib.Nifti1Image(tissue_masks["imat"].astype(np.uint8), ct_nii.affine, ct_nii.header),
             imat_out)
    saved_paths["imat"] = imat_out

    # Save low density muscle mask
    low_out = output_dir / "muscle_low_density.nii.gz"
    nib.save(nib.Nifti1Image(tissue_masks["muscle_low"].astype(np.uint8), ct_nii.affine, ct_nii.header),
             low_out)
    saved_paths["muscle_low"] = low_out

    # Save normal density muscle mask
    normal_out = output_dir / "muscle_normal.nii.gz"
    nib.save(nib.Nifti1Image(tissue_masks["muscle_normal"].astype(np.uint8), ct_nii.affine, ct_nii.header),
             normal_out)
    saved_paths["muscle_normal"] = normal_out

    # Save all muscle tissue mask
    all_out = output_dir / "muscle_all.nii.gz"
    nib.save(nib.Nifti1Image(tissue_masks["muscle_all"].astype(np.uint8), ct_nii.affine, ct_nii.header),
             all_out)
    saved_paths["muscle_all"] = all_out

    # Save combined classification mask
    labels = np.zeros(ct_data.shape, dtype=np.uint8)
    labels[tissue_masks["imat"]] = 1
    labels[tissue_masks["muscle_low"]] = 2
    labels[tissue_masks["muscle_normal"]] = 3

    classification_out = output_dir / "muscle_classification.nii.gz"
    nib.save(nib.Nifti1Image(labels, ct_nii.affine, ct_nii.header), classification_out)
    saved_paths["classification"] = classification_out

    logger.info(f"Saved {len(saved_paths)} muscle tissue masks to {output_dir}")

    return saved_paths


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 5:
        print("Usage: python muscle_analysis.py <ct_path> <envelope_path> <seg_dir> <cal_dir>")
        sys.exit(1)

    ct_path = Path(sys.argv[1])
    envelope_path = Path(sys.argv[2])
    seg_dir = Path(sys.argv[3])
    cal_dir = Path(sys.argv[4])

    result = analyze_muscle(ct_path, envelope_path, seg_dir, cal_dir)

    print(f"\nMuscle Analysis {'successful' if result.success else 'failed'}")
    print(f"Compartment volume: {result.compartment_volume_cm3:.1f} cm³")
    print(f"Muscle tissue volume: {result.muscle_tissue_volume_cm3:.1f} cm³")
    print(f"SMD: {result.muscle_SMD_mean_hu:.1f} HU (median: {result.muscle_SMD_median_hu:.1f})")
    print(f"Low density muscle: {result.muscle_low_density_percent:.1f}%")
    print(f"IMAT: {result.imat_percent:.1f}% ({result.imat_volume_cm3:.1f} cm³)")
    print(f"CSA: mean={result.muscle_CSA_mean_cm2:.1f} cm², max={result.muscle_CSA_max_cm2:.1f} cm²")
    print(f"L/R symmetry: {result.muscle_LR_symmetry_index:.2f}")
    print("\nQC messages:")
    for msg in result.qc_messages:
        print(f"  {msg}")
