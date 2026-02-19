#!/usr/bin/env python3
"""
vertebra_detection.py - Robust vertebra detection from TotalSegmentator outputs

TotalSegmentator often mislabels vertebrae when FOV is limited. This module
finds the two vertebrae with the largest voxel counts (which correspond to
vertebrae fully captured in FOV) and standardizes them to L1/L2 naming.

Key assumptions:
- The imaging FOV centers on L1-L2 vertebrae
- The two largest vertebrae by voxel count are the central FOV vertebrae
- Superior vertebra (higher Z in RAS) = L1, inferior = L2

Subject-specific overrides:
- sub-114: TotalSegmentator produces overlapping L1/L2 masks spanning multiple
  vertebrae. We combine L1+L2 and split by Z-centroid midpoint.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)

# Expected vertebra labels from TotalSegmentator (T10 to L4)
VERTEBRA_LABELS = [
    "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4"
]

# Minimum voxel count for a valid lumbar vertebra (prevents fragments)
# ~5 cm³ at 1mm³ resolution, or ~2000 voxels at 1.25mm resolution
MIN_VOXEL_COUNT = 2000

# Subjects requiring special handling due to TotalSegmentator mislabeling
# These subjects have L1/L2 masks that span multiple vertebrae
SUBJECTS_REQUIRING_Z_SPLIT = ["sub-114"]


@dataclass
class VertebraInfo:
    """Information about a detected vertebra."""
    original_label: str
    nifti_path: str  # String for JSON serialization
    voxel_count: int
    z_centroid: float
    volume_cm3: float


@dataclass
class VertebraDetectionResult:
    """Result of vertebra detection."""
    success: bool
    l1_info: Optional[VertebraInfo]
    l2_info: Optional[VertebraInfo]
    all_detected: List[VertebraInfo]
    qc_messages: List[str]


def _handle_z_split_subject(
    segmentations_dir: Path,
    subject_id: str
) -> VertebraDetectionResult:
    """
    Handle subjects where TotalSegmentator produces overlapping L1/L2 masks.

    For these subjects, we:
    1. Combine L1 + L2 + T11 masks (union) - T11 is included if available to capture
       more vertebral tissue for the upper region
    2. Split at the midpoint of L1/L2 Z-centroids
    3. Upper half becomes our "L1", lower half becomes our "L2"
    4. Save the split masks as temporary files for downstream processing

    Args:
        segmentations_dir: Directory with TotalSegmentator outputs
        subject_id: Subject identifier for logging

    Returns:
        VertebraDetectionResult with split vertebrae
    """
    qc_messages = [f"OVERRIDE: {subject_id} requires Z-split due to TotalSegmentator mislabeling"]

    l1_path = segmentations_dir / "vertebrae_L1.nii.gz"
    l2_path = segmentations_dir / "vertebrae_L2.nii.gz"
    t11_path = segmentations_dir / "vertebrae_T11.nii.gz"
    t12_path = segmentations_dir / "vertebrae_T12.nii.gz"

    if not l1_path.exists() or not l2_path.exists():
        qc_messages.append("ERROR: Cannot apply Z-split - L1 or L2 mask not found")
        return VertebraDetectionResult(
            success=False, l1_info=None, l2_info=None,
            all_detected=[], qc_messages=qc_messages
        )

    # Load L1 and L2 masks
    l1_nii = nib.load(l1_path)
    l2_nii = nib.load(l2_path)
    l1_data = l1_nii.get_fdata().astype(bool)
    l2_data = l2_nii.get_fdata().astype(bool)

    # Optionally include T11 and/or T12 if they exist (helps fill in upper region)
    extra_masks = []
    for extra_path, extra_name in [(t11_path, "T11"), (t12_path, "T12")]:
        if extra_path.exists():
            extra_nii = nib.load(extra_path)
            extra_data = extra_nii.get_fdata().astype(bool)
            extra_voxels = extra_data.sum()
            if extra_voxels > 1000:  # Only include if substantial
                extra_masks.append((extra_name, extra_data, extra_voxels))
                qc_messages.append(f"Including {extra_name} mask ({extra_voxels:,} voxels) in combined region")

    # Find Z-centroids
    l1_z_coords = np.where(l1_data)[2]
    l2_z_coords = np.where(l2_data)[2]

    if len(l1_z_coords) == 0 or len(l2_z_coords) == 0:
        qc_messages.append("ERROR: L1 or L2 mask is empty")
        return VertebraDetectionResult(
            success=False, l1_info=None, l2_info=None,
            all_detected=[], qc_messages=qc_messages
        )

    l1_centroid = np.mean(l1_z_coords)
    l2_centroid = np.mean(l2_z_coords)
    split_z = int((l1_centroid + l2_centroid) / 2)

    qc_messages.append(f"Original L1 centroid: z={l1_centroid:.1f}, L2 centroid: z={l2_centroid:.1f}")
    qc_messages.append(f"Splitting combined mask at z={split_z}")

    # Combine L1, L2, and optionally T11/T12
    combined = l1_data | l2_data
    for extra_name, extra_data, _ in extra_masks:
        combined = combined | extra_data

    # Upper half (z >= split_z) becomes new "L1"
    new_l1_data = combined.copy()
    new_l1_data[:, :, :split_z] = False

    # Lower half (z < split_z) becomes new "L2"
    new_l2_data = combined.copy()
    new_l2_data[:, :, split_z:] = False

    # Compute stats
    voxel_sizes = l1_nii.header.get_zooms()[:3]
    voxel_vol_cm3 = float(np.prod(voxel_sizes)) / 1000.0

    new_l1_voxels = int(new_l1_data.sum())
    new_l2_voxels = int(new_l2_data.sum())

    if new_l1_voxels == 0 or new_l2_voxels == 0:
        qc_messages.append("ERROR: Z-split resulted in empty mask")
        return VertebraDetectionResult(
            success=False, l1_info=None, l2_info=None,
            all_detected=[], qc_messages=qc_messages
        )

    new_l1_z = np.where(new_l1_data)[2]
    new_l2_z = np.where(new_l2_data)[2]

    new_l1_centroid = float(np.mean(new_l1_z))
    new_l2_centroid = float(np.mean(new_l2_z))

    # Save split masks to temporary files (overwrite originals for this session)
    # The pipeline will use these for vertebral body isolation
    split_l1_path = segmentations_dir / "vertebrae_L1_zsplit.nii.gz"
    split_l2_path = segmentations_dir / "vertebrae_L2_zsplit.nii.gz"

    split_l1_nii = nib.Nifti1Image(new_l1_data.astype(np.uint8), l1_nii.affine, l1_nii.header)
    split_l2_nii = nib.Nifti1Image(new_l2_data.astype(np.uint8), l2_nii.affine, l2_nii.header)

    nib.save(split_l1_nii, split_l1_path)
    nib.save(split_l2_nii, split_l2_path)

    qc_messages.append(f"Saved Z-split L1: {new_l1_voxels:,} voxels, {new_l1_voxels * voxel_vol_cm3:.1f} cm³")
    qc_messages.append(f"Saved Z-split L2: {new_l2_voxels:,} voxels, {new_l2_voxels * voxel_vol_cm3:.1f} cm³")

    # Create VertebraInfo for split masks
    l1_info = VertebraInfo(
        original_label="vertebrae_L1+L2_zsplit_upper",
        nifti_path=str(split_l1_path),
        voxel_count=new_l1_voxels,
        z_centroid=new_l1_centroid,
        volume_cm3=round(new_l1_voxels * voxel_vol_cm3, 2)
    )

    l2_info = VertebraInfo(
        original_label="vertebrae_L1+L2_zsplit_lower",
        nifti_path=str(split_l2_path),
        voxel_count=new_l2_voxels,
        z_centroid=new_l2_centroid,
        volume_cm3=round(new_l2_voxels * voxel_vol_cm3, 2)
    )

    qc_messages.append(f"MAPPING: Z-split upper -> L1 (z={new_l1_centroid:.1f})")
    qc_messages.append(f"MAPPING: Z-split lower -> L2 (z={new_l2_centroid:.1f})")

    return VertebraDetectionResult(
        success=True,
        l1_info=l1_info,
        l2_info=l2_info,
        all_detected=[l1_info, l2_info],
        qc_messages=qc_messages
    )


def detect_central_vertebrae(
    segmentations_dir: Path,
    min_voxel_count: int = MIN_VOXEL_COUNT,
    subject_id: Optional[str] = None
) -> VertebraDetectionResult:
    """
    Detect and identify the two central FOV vertebrae.

    Algorithm:
    1. Scan segmentations_dir for all vertebrae_*.nii.gz files (T10-L4 range)
    2. Load each and compute voxel count + Z-centroid
    3. Filter out fragments (below min voxel count)
    4. Select the two with the largest voxel counts
    5. Sort by Z-coordinate: superior = L1, inferior = L2

    For subjects in SUBJECTS_REQUIRING_Z_SPLIT (e.g., sub-114), we use a
    different approach: combine L1+L2 masks and split by Z-centroid midpoint.

    Args:
        segmentations_dir: Directory with TotalSegmentator outputs
        min_voxel_count: Minimum voxels for valid vertebra (filters fragments)
        subject_id: Optional subject identifier for subject-specific handling

    Returns:
        VertebraDetectionResult with detection info and QC messages
    """
    segmentations_dir = Path(segmentations_dir)

    # Check for subject-specific override
    if subject_id and subject_id in SUBJECTS_REQUIRING_Z_SPLIT:
        logger.info(f"Using Z-split approach for {subject_id}")
        return _handle_z_split_subject(segmentations_dir, subject_id)

    qc_messages = []
    detected_vertebrae = []

    # Step 1: Find all vertebra masks in expected range
    for label in VERTEBRA_LABELS:
        mask_path = segmentations_dir / f"{label}.nii.gz"
        if not mask_path.exists():
            continue

        try:
            nii = nib.load(mask_path)
            mask_data = nii.get_fdata().astype(bool)
            voxel_count = int(mask_data.sum())

            if voxel_count == 0:
                continue

            # Compute Z-centroid (mean Z index of all voxels)
            z_coords = np.where(mask_data)[2]
            z_centroid = float(np.mean(z_coords))

            # Compute volume
            voxel_sizes = nii.header.get_zooms()[:3]
            voxel_vol_cm3 = float(np.prod(voxel_sizes)) / 1000.0
            volume_cm3 = voxel_count * voxel_vol_cm3

            info = VertebraInfo(
                original_label=label,
                nifti_path=str(mask_path),
                voxel_count=voxel_count,
                z_centroid=z_centroid,
                volume_cm3=round(volume_cm3, 2)
            )
            detected_vertebrae.append(info)

            logger.debug(f"Found {label}: {voxel_count} voxels, z={z_centroid:.1f}")

        except Exception as e:
            logger.warning(f"Error loading {mask_path}: {e}")
            continue

    # Log all detected vertebrae
    qc_messages.append(f"Detected {len(detected_vertebrae)} vertebrae in FOV:")
    for v in sorted(detected_vertebrae, key=lambda x: x.z_centroid, reverse=True):
        qc_messages.append(f"  {v.original_label}: {v.voxel_count} voxels, "
                          f"z={v.z_centroid:.1f}, vol={v.volume_cm3:.1f} cm³")

    # Step 2: Filter out fragments (too small to be valid vertebrae)
    valid_vertebrae = [
        v for v in detected_vertebrae
        if v.voxel_count >= min_voxel_count
    ]

    excluded = len(detected_vertebrae) - len(valid_vertebrae)
    if excluded > 0:
        qc_messages.append(f"Excluded {excluded} vertebrae below minimum size "
                          f"({min_voxel_count} voxels)")

    if len(valid_vertebrae) < 2:
        qc_messages.append(f"ERROR: Only {len(valid_vertebrae)} valid vertebrae found, need 2")
        return VertebraDetectionResult(
            success=False,
            l1_info=None,
            l2_info=None,
            all_detected=detected_vertebrae,
            qc_messages=qc_messages
        )

    # Step 3: Select two largest by voxel count
    sorted_by_size = sorted(valid_vertebrae, key=lambda x: x.voxel_count, reverse=True)
    top_two = sorted_by_size[:2]

    # Step 4: Sort by Z-coordinate (higher Z = superior = L1)
    sorted_by_z = sorted(top_two, key=lambda x: x.z_centroid, reverse=True)
    l1_candidate = sorted_by_z[0]  # Superior
    l2_candidate = sorted_by_z[1]  # Inferior

    # Log the mapping
    qc_messages.append(f"MAPPING: {l1_candidate.original_label} -> L1 "
                      f"(superior, z={l1_candidate.z_centroid:.1f})")
    qc_messages.append(f"MAPPING: {l2_candidate.original_label} -> L2 "
                      f"(inferior, z={l2_candidate.z_centroid:.1f})")

    # Validate the mapping makes anatomical sense
    z_gap = l1_candidate.z_centroid - l2_candidate.z_centroid
    if z_gap < 5:  # Less than ~5 voxels separation
        qc_messages.append(f"WARNING: Small Z-gap between L1 and L2 ({z_gap:.1f} voxels)")

    # Check if original labels are unexpected
    expected_labels = ["vertebrae_L1", "vertebrae_L2", "vertebrae_T12", "vertebrae_L3"]
    if l1_candidate.original_label not in expected_labels:
        qc_messages.append(f"WARNING: L1 mapped from unexpected {l1_candidate.original_label}")
    if l2_candidate.original_label not in expected_labels:
        qc_messages.append(f"WARNING: L2 mapped from unexpected {l2_candidate.original_label}")

    return VertebraDetectionResult(
        success=True,
        l1_info=l1_candidate,
        l2_info=l2_candidate,
        all_detected=detected_vertebrae,
        qc_messages=qc_messages
    )


def save_detection_result(result: VertebraDetectionResult, output_path: Path):
    """
    Save detection result to JSON file.

    Args:
        result: VertebraDetectionResult to save
        output_path: Path for output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "success": result.success,
        "l1": asdict(result.l1_info) if result.l1_info else None,
        "l2": asdict(result.l2_info) if result.l2_info else None,
        "all_detected": [asdict(v) for v in result.all_detected],
        "qc_messages": result.qc_messages
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved vertebra detection result to {output_path}")


def standardize_vertebrae(
    detection_result: VertebraDetectionResult,
    output_dir: Path,
    isolate_body_func=None
) -> Dict[str, Path]:
    """
    Create standardized L1/L2 vertebral body files from detection result.

    Args:
        detection_result: Output from detect_central_vertebrae()
        output_dir: Directory for output files
        isolate_body_func: Optional function to isolate vertebral body.
                          If None, uses raw mask.

    Returns:
        Dictionary with paths to standardized files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    for vertebra, standardized_name in [
        (detection_result.l1_info, "L1"),
        (detection_result.l2_info, "L2")
    ]:
        if vertebra is None:
            continue

        nii = nib.load(vertebra.nifti_path)

        if isolate_body_func is not None:
            # Apply vertebral body isolation (remove posterior elements)
            body_nii = isolate_body_func(nii)
        else:
            body_nii = nii

        output_path = output_dir / f"{standardized_name}_body.nii.gz"
        nib.save(body_nii, output_path)

        outputs[standardized_name] = output_path
        logger.info(f"Saved {standardized_name}_body.nii.gz "
                   f"(from {vertebra.original_label}, {vertebra.voxel_count} voxels)")

    return outputs


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python vertebra_detection.py <segmentations_dir> [output_dir]")
        sys.exit(1)

    seg_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else seg_dir.parent / "vertebral_bodies"

    result = detect_central_vertebrae(seg_dir)

    print("\nVertebra Detection Result:")
    print(f"Success: {result.success}")

    if result.l1_info:
        print(f"L1: {result.l1_info.original_label} ({result.l1_info.voxel_count} voxels)")
    if result.l2_info:
        print(f"L2: {result.l2_info.original_label} ({result.l2_info.voxel_count} voxels)")

    print("\nQC Messages:")
    for msg in result.qc_messages:
        print(f"  {msg}")

    if result.success:
        save_detection_result(result, output_dir / "vertebra_detection.json")
        paths = standardize_vertebrae(result, output_dir)
        print(f"\nSaved standardized vertebrae to {output_dir}")
