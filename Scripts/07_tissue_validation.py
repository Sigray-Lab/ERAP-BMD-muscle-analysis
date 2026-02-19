#!/usr/bin/env python3
"""
tissue_validation.py - Validate HU-based tissue classification against TotalSegmentator

This module compares:
1. HU-based IMAT (from muscle envelope) vs TotalSegmentator intermuscular_fat
   - IMPORTANT: TotalSegmentator IMAT is restricted to the same muscle envelope mask
     used by the HU-based method to ensure fair comparison and avoid phantom artifacts

2. Muscle comparison is deferred (TotalSegmentator skeletal_muscle includes muscles
   outside the erector spinae compartment we're analyzing)

3. VAT/SAT analysis is commented out due to phantom tray artifacts (narrow FOV issue)

Outputs:
- Dice coefficients for spatial overlap (within muscle envelope)
- Volume correlations
- ts_imat_in_envelope_volume_cm3: TotalSegmentator IMAT restricted to muscle envelope
  (for longitudinal comparison alongside HU-based IMAT)
"""

import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.spatial.distance import dice as scipy_dice
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from tissue validation comparison."""
    success: bool
    subject_id: str
    session: str

    # IMAT comparison (both restricted to muscle envelope)
    imat_dice: float
    imat_hu_volume_cm3: float           # HU-based IMAT within envelope
    imat_ts_volume_cm3: float           # TS IMAT restricted to envelope (for comparison)
    imat_ts_in_envelope_volume_cm3: float  # Same as above, explicit name for results
    imat_volume_ratio: float

    # Raw TotalSegmentator IMAT (not restricted - for reference only)
    ts_imat_raw_volume_cm3: float

    # Muscle envelope info
    muscle_envelope_volume_cm3: float

    # QC flags
    imat_dice_acceptable: bool  # Dice > 0.5
    envelope_coverage_ok: bool  # TS IMAT has reasonable overlap with envelope

    # VAT/SAT volumes commented out due to phantom artifacts
    # These fields are kept for future use but set to 0
    ts_vat_volume_cm3: float  # NOT USED - phantom artifacts
    ts_sat_volume_cm3: float  # NOT USED - phantom artifacts

    qc_messages: List[str]


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Returns value between 0 (no overlap) and 1 (perfect overlap).
    """
    mask1_bool = mask1.astype(bool).flatten()
    mask2_bool = mask2.astype(bool).flatten()

    if not mask1_bool.any() and not mask2_bool.any():
        return 1.0  # Both empty = perfect match
    if not mask1_bool.any() or not mask2_bool.any():
        return 0.0  # One empty = no match

    # scipy's dice returns dissimilarity (1 - dice), so we compute 1 - result
    return 1.0 - scipy_dice(mask1_bool, mask2_bool)


def restrict_to_z_range(mask: np.ndarray, z_range: Tuple[int, int]) -> np.ndarray:
    """Restrict mask to specified z-range, zeroing out other slices."""
    z_min, z_max = z_range
    restricted = np.zeros_like(mask)
    z_max_clipped = min(z_max + 1, mask.shape[2])
    restricted[:, :, z_min:z_max_clipped] = mask[:, :, z_min:z_max_clipped]
    return restricted


def get_z_range_from_vertebrae(vb_dir: Path) -> Optional[Tuple[int, int]]:
    """Get z-range from L1 and L2 vertebral body masks."""
    l1_path = vb_dir / "L1_body.nii.gz"
    l2_path = vb_dir / "L2_body.nii.gz"

    if not l1_path.exists() or not l2_path.exists():
        return None

    l1_mask = nib.load(l1_path).get_fdata().astype(bool)
    l2_mask = nib.load(l2_path).get_fdata().astype(bool)

    l1_z = np.where(l1_mask.any(axis=(0, 1)))[0]
    l2_z = np.where(l2_mask.any(axis=(0, 1)))[0]

    if len(l1_z) == 0 or len(l2_z) == 0:
        return None

    z_min = min(l1_z.min(), l2_z.min())
    z_max = max(l1_z.max(), l2_z.max())

    return (int(z_min), int(z_max))


def validate_tissue_segmentation(
    derived_dir: Path,
    subject_id: str,
    session: str
) -> ValidationResult:
    """
    Validate HU-based tissue classification against TotalSegmentator tissue_4_types.

    IMPORTANT: TotalSegmentator IMAT (intermuscular_fat) is restricted to the same
    muscle envelope mask used by the HU-based method. This ensures:
    1. Fair comparison (same anatomical region)
    2. Avoids phantom tray artifacts that appear in raw TS masks

    Args:
        derived_dir: Directory containing session-specific derived data
        subject_id: Subject identifier
        session: Session name

    Returns:
        ValidationResult with comparison metrics
    """
    qc_messages = []

    # Initialize result with defaults
    result = ValidationResult(
        success=False,
        subject_id=subject_id,
        session=session,
        imat_dice=np.nan,
        imat_hu_volume_cm3=0.0,
        imat_ts_volume_cm3=0.0,
        imat_ts_in_envelope_volume_cm3=0.0,
        imat_volume_ratio=np.nan,
        ts_imat_raw_volume_cm3=0.0,
        muscle_envelope_volume_cm3=0.0,
        imat_dice_acceptable=False,
        envelope_coverage_ok=True,
        ts_vat_volume_cm3=0.0,  # Not used - phantom artifacts
        ts_sat_volume_cm3=0.0,  # Not used - phantom artifacts
        qc_messages=qc_messages
    )

    # Get paths
    seg_dir = derived_dir / "segmentations"
    muscle_masks_dir = derived_dir / "muscle_masks"
    vb_dir = derived_dir / "vertebral_bodies"

    # Get z-range from vertebrae
    z_range = get_z_range_from_vertebrae(vb_dir)
    if z_range is None:
        qc_messages.append("ERROR: Could not determine z-range from vertebrae")
        return result

    logger.info(f"Using z-range: {z_range[0]} to {z_range[1]}")

    # Load muscle envelope - this is the key mask for restricting comparison
    envelope_path = muscle_masks_dir / "muscle_envelope.nii.gz"
    if not envelope_path.exists():
        # Try alternative location
        envelope_path = derived_dir / "muscle_compartment.nii.gz"

    if not envelope_path.exists():
        qc_messages.append("ERROR: Muscle envelope mask not found")
        return result

    envelope_nii = nib.load(envelope_path)
    envelope_mask = envelope_nii.get_fdata().astype(bool)

    # Get voxel sizes from envelope
    voxel_sizes = envelope_nii.header.get_zooms()[:3]
    voxel_vol_cm3 = np.prod(voxel_sizes) / 1000.0

    # Restrict envelope to L1-L2 z-range
    envelope_restricted = restrict_to_z_range(envelope_mask, z_range)
    result.muscle_envelope_volume_cm3 = envelope_restricted.sum() * voxel_vol_cm3

    logger.info(f"Muscle envelope volume (L1-L2): {result.muscle_envelope_volume_cm3:.2f} cm³")

    # ===== IMAT Comparison (within muscle envelope) =====
    hu_imat_path = muscle_masks_dir / "muscle_imat.nii.gz"
    ts_imat_path = seg_dir / "intermuscular_fat.nii.gz"

    if hu_imat_path.exists() and ts_imat_path.exists():
        hu_imat = nib.load(hu_imat_path).get_fdata().astype(bool)
        ts_imat_raw = nib.load(ts_imat_path).get_fdata().astype(bool)

        # Restrict both to L1-L2 z-range
        hu_imat_restricted = restrict_to_z_range(hu_imat, z_range)
        ts_imat_raw_restricted = restrict_to_z_range(ts_imat_raw, z_range)

        # CRITICAL: Restrict TotalSegmentator IMAT to muscle envelope
        # This ensures we're comparing the same anatomical region
        ts_imat_in_envelope = ts_imat_raw_restricted & envelope_restricted

        # HU-based IMAT is already within envelope by definition
        result.imat_hu_volume_cm3 = hu_imat_restricted.sum() * voxel_vol_cm3

        # TS IMAT restricted to envelope (for fair comparison)
        result.imat_ts_volume_cm3 = ts_imat_in_envelope.sum() * voxel_vol_cm3
        result.imat_ts_in_envelope_volume_cm3 = result.imat_ts_volume_cm3

        # Raw TS IMAT (for reference - includes phantom artifacts)
        result.ts_imat_raw_volume_cm3 = ts_imat_raw_restricted.sum() * voxel_vol_cm3

        if result.imat_ts_volume_cm3 > 0:
            result.imat_volume_ratio = result.imat_hu_volume_cm3 / result.imat_ts_volume_cm3
        else:
            result.imat_volume_ratio = np.nan

        # Compute Dice within envelope
        result.imat_dice = compute_dice(hu_imat_restricted, ts_imat_in_envelope)
        result.imat_dice_acceptable = result.imat_dice >= 0.5

        # Check envelope coverage
        if result.ts_imat_raw_volume_cm3 > 0:
            coverage = result.imat_ts_volume_cm3 / result.ts_imat_raw_volume_cm3
            result.envelope_coverage_ok = coverage > 0.3  # At least 30% should be in envelope
            if not result.envelope_coverage_ok:
                qc_messages.append(
                    f"WARNING: Only {coverage:.1%} of TS IMAT is within muscle envelope. "
                    "Raw TS mask may include phantom or other artifacts."
                )

        logger.info(f"IMAT (within envelope): Dice={result.imat_dice:.3f}, "
                    f"HU vol={result.imat_hu_volume_cm3:.2f} cm³, "
                    f"TS vol={result.imat_ts_volume_cm3:.2f} cm³")
        logger.info(f"TS IMAT raw (L1-L2, no envelope restriction): {result.ts_imat_raw_volume_cm3:.2f} cm³")

        if not result.imat_dice_acceptable:
            qc_messages.append(f"INFO: Low IMAT Dice coefficient = {result.imat_dice:.3f}")
    else:
        if not hu_imat_path.exists():
            qc_messages.append("INFO: HU-based IMAT mask not found")
        if not ts_imat_path.exists():
            qc_messages.append("INFO: TotalSegmentator IMAT mask not found (tissue_4_types not run?)")

    # ===== VAT/SAT Analysis - COMMENTED OUT due to phantom artifacts =====
    # The narrow FOV causes TotalSegmentator to pick up phantom calibration
    # rods as fat tissue. These volumes are unreliable and set to 0.
    #
    # tissue_files = {
    #     "ts_vat_volume_cm3": "torso_fat.nii.gz",
    #     "ts_sat_volume_cm3": "subcutaneous_fat.nii.gz",
    # }
    #
    # for attr, filename in tissue_files.items():
    #     path = seg_dir / filename
    #     if path.exists():
    #         mask = nib.load(path).get_fdata().astype(bool)
    #         restricted = restrict_to_z_range(mask, z_range)
    #         volume = restricted.sum() * voxel_vol_cm3
    #         setattr(result, attr, volume)
    #         logger.info(f"{filename}: {volume:.2f} cm³ (L1-L2 restricted)")
    #     else:
    #         logger.warning(f"Tissue mask not found: {filename}")
    #
    # NOTE: VAT/SAT values remain at 0.0 - phantom artifacts need manual correction

    # Determine overall success
    result.success = not np.isnan(result.imat_dice)

    result.qc_messages = qc_messages
    return result


def validate_all_subjects(
    derived_data_dir: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Run validation on all subjects and aggregate results.

    Args:
        derived_data_dir: Root directory containing subject folders
        output_path: Path for output CSV

    Returns:
        DataFrame with validation results
    """
    all_results = []

    # Find all subject directories
    subject_dirs = sorted([d for d in derived_data_dir.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])

    logger.info(f"Found {len(subject_dirs)} subject directories")

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        for session in ["ses-Baseline", "ses-Followup"]:
            session_dir = subject_dir / session

            if not session_dir.exists():
                continue

            logger.info(f"\nValidating {subject_id} / {session}")

            try:
                result = validate_tissue_segmentation(
                    session_dir, subject_id, session
                )
                all_results.append(asdict(result))

            except Exception as e:
                logger.error(f"Validation failed for {subject_id}/{session}: {e}")
                all_results.append({
                    "success": False,
                    "subject_id": subject_id,
                    "session": session,
                    "qc_messages": [f"ERROR: {e}"]
                })

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save CSV
    df.to_csv(output_path, index=False)
    logger.info(f"\nValidation results saved to {output_path}")

    # Print summary
    print_validation_summary(df)

    return df


def print_validation_summary(df: pd.DataFrame):
    """Print summary statistics from validation results."""
    print("\n" + "=" * 60)
    print("TISSUE VALIDATION SUMMARY (IMAT within Muscle Envelope)")
    print("=" * 60)

    n_total = len(df)
    n_success = df["success"].sum() if "success" in df.columns else 0

    print(f"\nTotal sessions validated: {n_total}")
    print(f"Successful validations: {n_success}")

    if "imat_dice" in df.columns:
        imat_dice = df["imat_dice"].dropna()
        if len(imat_dice) > 0:
            print(f"\nIMAT Dice Coefficient (within envelope):")
            print(f"  Mean: {imat_dice.mean():.3f}")
            print(f"  Median: {imat_dice.median():.3f}")
            print(f"  Min: {imat_dice.min():.3f}")
            print(f"  Max: {imat_dice.max():.3f}")

    if "imat_hu_volume_cm3" in df.columns:
        hu_vol = df["imat_hu_volume_cm3"].dropna()
        ts_vol = df["imat_ts_in_envelope_volume_cm3"].dropna()
        if len(hu_vol) > 0:
            print(f"\nIMAT Volume Comparison:")
            print(f"  HU-based:    {hu_vol.mean():.1f} ± {hu_vol.std():.1f} cm³")
            print(f"  TS (in env): {ts_vol.mean():.1f} ± {ts_vol.std():.1f} cm³")

    if "imat_dice_acceptable" in df.columns:
        n_acceptable = df["imat_dice_acceptable"].sum()
        n_total_imat = len(df["imat_dice_acceptable"].dropna())
        if n_total_imat > 0:
            print(f"\nIMAT Dice >= 0.5: {n_acceptable}/{n_total_imat} "
                  f"({100*n_acceptable/n_total_imat:.0f}%)")


def save_validation_results(result: ValidationResult, output_path: Path):
    """Save single validation result to JSON."""
    output_dict = asdict(result)

    # Convert numpy types to Python native types for JSON serialization
    for key, value in output_dict.items():
        if isinstance(value, (np.floating, float)):
            if np.isnan(value):
                output_dict[key] = None
            else:
                output_dict[key] = float(value)
        elif isinstance(value, (np.bool_, bool)):
            output_dict[key] = bool(value)
        elif isinstance(value, np.integer):
            output_dict[key] = int(value)

    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    logger.info(f"Validation results saved to {output_path}")


def normalize_ct_window(ct_data: np.ndarray,
                        window_center: float = 40,
                        window_width: float = 400) -> np.ndarray:
    """Apply CT windowing for visualization."""
    low = window_center - window_width / 2
    high = window_center + window_width / 2
    normalized = np.clip(ct_data, low, high)
    normalized = (normalized - low) / (high - low)
    return normalized


def create_tissue_validation_figure(
    ct_path: Path,
    derived_dir: Path,
    output_path: Path,
    result: ValidationResult
):
    """
    Generate QC visualization comparing HU-based vs TotalSegmentator IMAT.

    Shows comparison within the muscle envelope mask to ensure fair comparison.

    Creates a 1x3 figure:
    - Col 1: HU-based IMAT (within envelope)
    - Col 2: TotalSegmentator IMAT (restricted to envelope)
    - Col 3: Overlap comparison with Dice

    Args:
        ct_path: Path to CT NIfTI
        derived_dir: Directory with derived data
        output_path: Output path for PNG
        result: ValidationResult with metrics to display
    """
    seg_dir = derived_dir / "segmentations"
    muscle_masks_dir = derived_dir / "muscle_masks"
    vb_dir = derived_dir / "vertebral_bodies"

    # Load CT
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()

    # Get z-range from vertebrae for consistent slice selection
    z_range = get_z_range_from_vertebrae(vb_dir)
    if z_range is None:
        logger.warning("Could not determine z-range for QC figure")
        return

    z_mid = (z_range[0] + z_range[1]) // 2

    # Load muscle envelope
    envelope_path = muscle_masks_dir / "muscle_envelope.nii.gz"
    if not envelope_path.exists():
        envelope_path = derived_dir / "muscle_compartment.nii.gz"

    if not envelope_path.exists():
        logger.warning("Muscle envelope not found for QC figure")
        return

    envelope_mask = nib.load(envelope_path).get_fdata().astype(bool)
    envelope_restricted = restrict_to_z_range(envelope_mask, z_range)

    # Load masks
    hu_imat_path = muscle_masks_dir / "muscle_imat.nii.gz"
    ts_imat_path = seg_dir / "intermuscular_fat.nii.gz"

    # Initialize masks as empty
    hu_imat = np.zeros(ct_data.shape, dtype=bool)
    ts_imat_in_envelope = np.zeros(ct_data.shape, dtype=bool)

    if hu_imat_path.exists():
        hu_imat = nib.load(hu_imat_path).get_fdata().astype(bool)
        hu_imat = restrict_to_z_range(hu_imat, z_range)

    if ts_imat_path.exists():
        ts_imat_raw = nib.load(ts_imat_path).get_fdata().astype(bool)
        ts_imat_raw = restrict_to_z_range(ts_imat_raw, z_range)
        # Restrict to envelope
        ts_imat_in_envelope = ts_imat_raw & envelope_restricted

    # Create figure - single row for IMAT comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ct_slice = normalize_ct_window(ct_data[:, :, z_mid], window_center=40, window_width=400)

    # Colors for overlays
    hu_color = [1.0, 0.0, 0.0, 0.5]        # Red for HU-based
    ts_color = [0.0, 0.0, 1.0, 0.5]        # Blue for TotalSegmentator
    overlap_color = [0.5, 0.0, 0.5, 0.6]   # Purple for overlap
    hu_only_color = [1.0, 0.5, 0.0, 0.5]   # Orange for HU-only
    ts_only_color = [0.0, 1.0, 1.0, 0.5]   # Cyan for TS-only
    envelope_color = [0.0, 1.0, 0.0, 0.3]  # Green for envelope outline

    # Get slices
    hu_imat_slice = hu_imat[:, :, z_mid]
    ts_imat_slice = ts_imat_in_envelope[:, :, z_mid]
    envelope_slice = envelope_restricted[:, :, z_mid]

    # IMAT - HU-based
    axes[0].imshow(ct_slice.T, cmap='gray', origin='lower')
    overlay = np.zeros((*ct_slice.shape, 4))
    overlay[hu_imat_slice, :] = hu_color
    axes[0].imshow(overlay.transpose(1, 0, 2), origin='lower')
    axes[0].contour(envelope_slice.T, colors='lime', linewidths=1, linestyles='dashed')
    axes[0].set_title(f"HU-based IMAT\n({result.imat_hu_volume_cm3:.1f} cm³)")

    # IMAT - TotalSegmentator (restricted to envelope)
    axes[1].imshow(ct_slice.T, cmap='gray', origin='lower')
    overlay = np.zeros((*ct_slice.shape, 4))
    overlay[ts_imat_slice, :] = ts_color
    axes[1].imshow(overlay.transpose(1, 0, 2), origin='lower')
    axes[1].contour(envelope_slice.T, colors='lime', linewidths=1, linestyles='dashed')
    axes[1].set_title(f"TotalSeg IMAT (in envelope)\n({result.imat_ts_volume_cm3:.1f} cm³)")

    # IMAT - Overlap comparison
    axes[2].imshow(ct_slice.T, cmap='gray', origin='lower')
    overlap_mask = hu_imat_slice & ts_imat_slice
    hu_only_mask = hu_imat_slice & ~ts_imat_slice
    ts_only_mask = ts_imat_slice & ~hu_imat_slice
    overlay = np.zeros((*ct_slice.shape, 4))
    overlay[overlap_mask, :] = overlap_color
    overlay[hu_only_mask, :] = hu_only_color
    overlay[ts_only_mask, :] = ts_only_color
    axes[2].imshow(overlay.transpose(1, 0, 2), origin='lower')
    axes[2].contour(envelope_slice.T, colors='lime', linewidths=1, linestyles='dashed')
    dice_str = f"{result.imat_dice:.3f}" if not np.isnan(result.imat_dice) else "N/A"
    axes[2].set_title(f"IMAT Overlap (Dice={dice_str})\nPurple=both, Orange=HU-only, Cyan=TS-only")

    # Add labels
    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    # Add legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, label='HU-based IMAT'),
        Patch(facecolor='blue', alpha=0.5, label='TotalSeg IMAT'),
        Patch(facecolor='purple', alpha=0.6, label='Overlap'),
        Patch(facecolor='orange', alpha=0.5, label='HU-only'),
        Patch(facecolor='cyan', alpha=0.5, label='TS-only'),
        Patch(facecolor='none', edgecolor='lime', linestyle='--', label='Muscle envelope'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))

    # Add subject/session info
    fig.suptitle(f"IMAT Validation (within envelope): {result.subject_id} / {result.session} (z={z_mid})",
                 fontsize=14, fontweight='bold')

    # Add info text
    info_text = (f"Envelope vol: {result.muscle_envelope_volume_cm3:.1f} cm³\n"
                 f"TS IMAT raw: {result.ts_imat_raw_volume_cm3:.1f} cm³")
    fig.text(0.02, 0.02, info_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Add QC warnings if any
    if result.qc_messages:
        qc_text = "\n".join(result.qc_messages[:3])
        fig.text(0.02, 0.15, qc_text, fontsize=8, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 0.95, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved tissue validation QC figure: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate HU-based tissue classification against TotalSegmentator"
    )

    parser.add_argument("--derived-dir", type=Path, required=True,
                        help="Path to DerivedData directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for output CSV")
    parser.add_argument("--subject", type=str, default=None,
                        help="Process only this subject")
    parser.add_argument("--session", type=str, default=None,
                        help="Process only this session")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.subject and args.session:
        # Single subject/session
        session_dir = args.derived_dir / args.subject / args.session

        if not session_dir.exists():
            print(f"Error: Session directory not found: {session_dir}")
            exit(1)

        result = validate_tissue_segmentation(
            session_dir, args.subject, args.session
        )

        print(f"\nValidation {'successful' if result.success else 'failed'}")
        print(f"IMAT Dice (in envelope): {result.imat_dice:.3f}")
        print(f"IMAT HU-based: {result.imat_hu_volume_cm3:.1f} cm³")
        print(f"IMAT TS (in envelope): {result.imat_ts_volume_cm3:.1f} cm³")
        print(f"IMAT TS raw (L1-L2): {result.ts_imat_raw_volume_cm3:.1f} cm³")

        print("\nQC messages:")
        for msg in result.qc_messages:
            print(f"  {msg}")

        # Save JSON
        json_path = args.output.with_suffix(".json")
        save_validation_results(result, json_path)

    else:
        # All subjects
        validate_all_subjects(args.derived_dir, args.output)
