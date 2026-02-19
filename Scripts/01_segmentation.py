#!/usr/bin/env python3
"""
segmentation.py - TotalSegmentator and derived mask generation

This module handles:
1. Running TotalSegmentator with fallback strategies
2. Isolating vertebral bodies from full vertebra masks
3. Creating muscle compartment envelopes from erector spinae masks
"""

import subprocess
import logging
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import (
    binary_closing, binary_fill_holes, distance_transform_edt,
    label, binary_erosion, generate_binary_structure
)
from skimage.morphology import ball

logger = logging.getLogger(__name__)


def check_totalsegmentator_output_exists(output_dir: Path, task: str = "roi_subset") -> bool:
    """
    Check if TotalSegmentator output already exists.

    Args:
        output_dir: Directory to check for segmentation outputs
        task: "roi_subset", "total", or "tissue_4_types"

    Returns:
        True if required outputs exist, False otherwise
    """
    if task == "roi_subset":
        # Check for at least 2 vertebrae and muscle masks
        vertebrae_found = 0
        for label in ["vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
                      "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4"]:
            if (output_dir / f"{label}.nii.gz").exists():
                vertebrae_found += 1

        muscle_exists = (
            (output_dir / "autochthon_left.nii.gz").exists() or
            (output_dir / "erector_spinae_left.nii.gz").exists()
        )

        return vertebrae_found >= 2 and muscle_exists

    elif task == "total":
        # Check for adipose masks (legacy task)
        return (
            (output_dir / "torso_fat.nii.gz").exists() or
            (output_dir / "body_trunc.nii.gz").exists() or
            (output_dir / "fat_visceral.nii.gz").exists()
        )

    elif task == "tissue_4_types":
        # Check for tissue_4_types outputs (VAT, SAT, muscle, IMAT)
        required_files = [
            "torso_fat.nii.gz",           # VAT
            "subcutaneous_fat.nii.gz",    # SAT
            "skeletal_muscle.nii.gz",     # Muscle
            "intermuscular_fat.nii.gz"    # IMAT
        ]
        return all((output_dir / f).exists() for f in required_files)

    return False


def run_totalsegmentator(ct_path: Path, output_dir: Path, task: str = "roi_subset",
                         force: bool = False) -> bool:
    """
    Run TotalSegmentator with fallback strategy.

    Args:
        ct_path: Path to input CT NIfTI file
        output_dir: Directory for segmentation outputs
        task: "roi_subset" for specific structures, "total" for full segmentation
        force: If True, run even if output exists

    Returns:
        True if successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if output already exists
    if not force and check_totalsegmentator_output_exists(output_dir, task):
        logger.info(f"TotalSegmentator {task} output already exists, skipping")
        return True

    if task == "roi_subset":
        # Try specific ROI subset first (fastest)
        # Include T10-L4 for robust vertebra detection (TotalSegmentator often mislabels)
        roi_structures = [
            "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
            "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4",
            "autochthon_left", "autochthon_right"
        ]
        cmd = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(output_dir),
            "--roi_subset", *roi_structures
        ]

        logger.info(f"Running TotalSegmentator with roi_subset: {roi_structures}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                logger.info("TotalSegmentator roi_subset completed successfully")
                return True
            else:
                logger.warning(f"roi_subset failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("TotalSegmentator roi_subset timed out")
        except Exception as e:
            logger.warning(f"roi_subset error: {e}")

        # Fallback to fast mode
        logger.info("Falling back to --fast mode")
        cmd = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(output_dir),
            "--fast"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if result.returncode == 0:
                logger.info("TotalSegmentator --fast completed successfully")
                return True
            else:
                logger.warning(f"--fast failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("TotalSegmentator --fast timed out")
        except Exception as e:
            logger.warning(f"--fast error: {e}")

        # Final fallback: full resolution
        logger.info("Falling back to full resolution")
        cmd = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(output_dir)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                logger.info("TotalSegmentator full resolution completed successfully")
                return True
            else:
                logger.error(f"Full resolution failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Full resolution error: {e}")
            return False

    elif task == "total":
        # Full segmentation for adipose tissue (legacy - prefer tissue_4_types)
        cmd = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(output_dir),
            "--task", "total"
        ]

        logger.info("Running TotalSegmentator with task=total for adipose")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                logger.info("TotalSegmentator task=total completed successfully")
                return True
            else:
                logger.error(f"task=total failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"task=total error: {e}")
            return False

    elif task == "tissue_4_types":
        # Tissue composition segmentation: VAT, SAT, muscle, IMAT
        # NOTE: This task requires a TotalSegmentator license. Activate with:
        #   totalseg_set_license -l <your_license_key>
        cmd = [
            "TotalSegmentator",
            "-i", str(ct_path),
            "-o", str(output_dir),
            "--task", "tissue_4_types"
        ]

        logger.info("Running TotalSegmentator with task=tissue_4_types for VAT/SAT/muscle/IMAT")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                logger.info("TotalSegmentator task=tissue_4_types completed successfully")
                return True
            else:
                # Check for license error
                if "license" in result.stderr.lower() or "academic" in result.stderr.lower():
                    logger.error(
                        "task=tissue_4_types requires a TotalSegmentator license. "
                        "Activate with: totalseg_set_license -l <your_license_key>"
                    )
                else:
                    logger.error(f"task=tissue_4_types failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("TotalSegmentator task=tissue_4_types timed out")
            return False
        except Exception as e:
            logger.error(f"task=tissue_4_types error: {e}")
            return False

    return False


def isolate_vertebral_body(vertebra_nii: nib.Nifti1Image,
                           endplate_exclude_percent: float = 10.0) -> nib.Nifti1Image:
    """
    Isolate vertebral body (centrum) from full vertebra mask.

    TotalSegmentator includes posterior elements (pedicles, laminae, spinous process)
    which bias trabecular BMD measurements. This function isolates just the body.

    Approach:
    1. Per axial slice: keep only the anterior (largest) connected component
    2. Exclude endplates (top/bottom portion of vertebral height)

    Args:
        vertebra_nii: NIfTI image of full vertebra mask
        endplate_exclude_percent: Percentage of slices to exclude from top/bottom

    Returns:
        NIfTI image of isolated vertebral body
    """
    mask = vertebra_nii.get_fdata().astype(bool)
    voxel_sizes = vertebra_nii.header.get_zooms()

    # Find z-range of vertebra
    z_indices = np.where(mask.any(axis=(0, 1)))[0]
    if len(z_indices) == 0:
        logger.warning("Empty vertebra mask")
        return nib.Nifti1Image(mask.astype(np.uint8), vertebra_nii.affine, vertebra_nii.header)

    z_min, z_max = z_indices.min(), z_indices.max()
    z_height = z_max - z_min + 1

    # Calculate endplate exclusion
    exclude_slices = int(z_height * endplate_exclude_percent / 100)
    z_start = z_min + exclude_slices
    z_end = z_max - exclude_slices

    if z_start >= z_end:
        logger.warning("Endplate exclusion too aggressive, using minimal exclusion")
        z_start = z_min + 1
        z_end = z_max - 1

    # Process each axial slice
    body_mask = np.zeros_like(mask, dtype=bool)

    for z in range(z_start, z_end + 1):
        slice_mask = mask[:, :, z]
        if not slice_mask.any():
            continue

        # Label connected components
        labeled, num_features = label(slice_mask)
        if num_features == 0:
            continue

        # Find the anterior component (vertebral body is anterior to posterior elements)
        # In standard orientation, anterior is toward higher y values (patient facing up)
        # We'll use the largest component as a heuristic for the body
        component_sizes = []
        component_centroids_y = []

        for comp_id in range(1, num_features + 1):
            comp_mask = labeled == comp_id
            component_sizes.append(comp_mask.sum())
            y_coords = np.where(comp_mask)[1]
            component_centroids_y.append(y_coords.mean() if len(y_coords) > 0 else 0)

        if not component_sizes:
            continue

        # Typically the vertebral body is the largest component
        # If there are multiple large components, prefer the one with higher y (more anterior)
        largest_idx = np.argmax(component_sizes)

        # Keep the largest component as the body
        body_mask[:, :, z] = (labeled == (largest_idx + 1))

    # Clean up: remove small disconnected regions in 3D
    labeled_3d, num_3d = label(body_mask)
    if num_3d > 1:
        # Keep only the largest 3D component
        sizes_3d = [np.sum(labeled_3d == i) for i in range(1, num_3d + 1)]
        largest_3d = np.argmax(sizes_3d) + 1
        body_mask = labeled_3d == largest_3d

    logger.info(f"Isolated vertebral body: {body_mask.sum()} voxels "
                f"(z-range: {z_start}-{z_end}, excluded {exclude_slices} endplate slices each side)")

    return nib.Nifti1Image(body_mask.astype(np.uint8), vertebra_nii.affine, vertebra_nii.header)


def create_muscle_envelope(erector_left_nii: nib.Nifti1Image,
                           erector_right_nii: nib.Nifti1Image,
                           l1_nii: nib.Nifti1Image,
                           l2_nii: nib.Nifti1Image,
                           closing_radius_mm: float = 5.0) -> nib.Nifti1Image:
    """
    Create fascial envelope for muscle compartment analysis.

    TotalSegmentator segments muscle tissue, excluding fat voxels. This creates
    a "Swiss cheese" effect where IMAT (intermuscular adipose) appears as holes.
    To measure IMAT, we need to create a fascial envelope that includes the fat.

    Approach:
    1. Combine left + right erector spinae masks
    2. Morphological closing to fill gaps
    3. 2D hole filling per slice
    4. Restrict to L1-L2 z-range

    Args:
        erector_left_nii: Left erector spinae mask
        erector_right_nii: Right erector spinae mask
        l1_nii: L1 vertebra mask (for z-range)
        l2_nii: L2 vertebra mask (for z-range)
        closing_radius_mm: Radius for morphological closing in mm

    Returns:
        NIfTI image of filled muscle compartment envelope
    """
    left_mask = erector_left_nii.get_fdata().astype(bool)
    right_mask = erector_right_nii.get_fdata().astype(bool)
    l1_mask = l1_nii.get_fdata().astype(bool)
    l2_mask = l2_nii.get_fdata().astype(bool)

    voxel_sizes = np.array(erector_left_nii.header.get_zooms()[:3])

    # Combine left and right
    combined = left_mask | right_mask

    if not combined.any():
        logger.warning("Empty erector spinae masks")
        return nib.Nifti1Image(combined.astype(np.uint8),
                               erector_left_nii.affine, erector_left_nii.header)

    # Determine z-range from L1 and L2
    l1_z = np.where(l1_mask.any(axis=(0, 1)))[0]
    l2_z = np.where(l2_mask.any(axis=(0, 1)))[0]

    if len(l1_z) == 0 or len(l2_z) == 0:
        logger.warning("L1 or L2 mask empty, using full z-range of erector spinae")
        z_min = np.where(combined.any(axis=(0, 1)))[0].min()
        z_max = np.where(combined.any(axis=(0, 1)))[0].max()
    else:
        # L1 is superior to L2 in vertebral numbering
        z_min = min(l1_z.min(), l2_z.min())
        z_max = max(l1_z.max(), l2_z.max())

    logger.info(f"Muscle compartment z-range: {z_min} to {z_max}")

    # Create structuring element for closing
    # Convert radius from mm to voxels
    radius_voxels = (closing_radius_mm / voxel_sizes).astype(int)
    radius_voxels = np.maximum(radius_voxels, 1)  # At least 1 voxel

    # Create 3D ball structuring element
    struct_radius = int(np.mean(radius_voxels))
    struct = ball(struct_radius)

    # Morphological closing
    logger.info(f"Applying morphological closing with radius {struct_radius} voxels")
    closed = binary_closing(combined, structure=struct)

    # 2D hole filling per slice
    filled = np.zeros_like(closed, dtype=bool)
    for z in range(closed.shape[2]):
        if closed[:, :, z].any():
            filled[:, :, z] = binary_fill_holes(closed[:, :, z])

    # Restrict to L1-L2 z-range
    envelope = np.zeros_like(filled, dtype=bool)
    envelope[:, :, z_min:z_max+1] = filled[:, :, z_min:z_max+1]

    logger.info(f"Muscle envelope created: {envelope.sum()} voxels "
                f"(original combined: {combined.sum()}, after closing: {closed.sum()})")

    return nib.Nifti1Image(envelope.astype(np.uint8),
                           erector_left_nii.affine, erector_left_nii.header)


def get_vertebra_z_range(l1_nii: nib.Nifti1Image,
                         l2_nii: nib.Nifti1Image) -> tuple:
    """
    Get the combined z-range (in voxel indices) covered by L1 and L2 vertebrae.

    Args:
        l1_nii: L1 vertebra mask
        l2_nii: L2 vertebra mask

    Returns:
        Tuple of (z_min, z_max) voxel indices
    """
    l1_mask = l1_nii.get_fdata().astype(bool)
    l2_mask = l2_nii.get_fdata().astype(bool)

    l1_z = np.where(l1_mask.any(axis=(0, 1)))[0]
    l2_z = np.where(l2_mask.any(axis=(0, 1)))[0]

    if len(l1_z) == 0 or len(l2_z) == 0:
        raise ValueError("L1 or L2 mask is empty")

    z_min = min(l1_z.min(), l2_z.min())
    z_max = max(l1_z.max(), l2_z.max())

    return z_min, z_max


def process_segmentation(ct_path: Path, derived_dir: Path, force: bool = False) -> dict:
    """
    Run complete segmentation pipeline for a single CT scan.

    Args:
        ct_path: Path to input CT NIfTI file
        derived_dir: Directory for derived data outputs
        force: If True, re-run TotalSegmentator even if output exists

    Returns:
        Dictionary with paths to generated files and status
    """
    results = {
        "success": False,
        "segmentations_dir": None,
        "vertebral_bodies_dir": None,
        "muscle_envelope_path": None,
        "l1_z_range": None,
        "l2_z_range": None,
        "vertebra_detection": None,
        "errors": []
    }

    # Create output directories
    seg_dir = derived_dir / "segmentations"
    vb_dir = derived_dir / "vertebral_bodies"
    seg_dir.mkdir(parents=True, exist_ok=True)
    vb_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run TotalSegmentator for spine and muscle
    logger.info(f"Processing: {ct_path}")

    if not run_totalsegmentator(ct_path, seg_dir, task="roi_subset", force=force):
        results["errors"].append("TotalSegmentator roi_subset failed")
        return results

    results["segmentations_dir"] = seg_dir

    # Run tissue_4_types for VAT/SAT/muscle/IMAT segmentation
    # This replaces the old task=total and provides learned tissue masks
    if not run_totalsegmentator(ct_path, seg_dir, task="tissue_4_types", force=force):
        logger.warning("TotalSegmentator task=tissue_4_types failed, VAT/IMAT analysis may be limited")
        results["errors"].append("TotalSegmentator task=tissue_4_types failed (non-fatal)")
    else:
        results["tissue_4_types_available"] = True

    # Step 2: Robust vertebra detection and body isolation
    # TotalSegmentator often mislabels vertebrae - we find the two largest and assign by Z-position
    try:
        from utils.vertebra_detection import (
            detect_central_vertebrae, standardize_vertebrae, save_detection_result
        )

        # Extract subject_id from path (e.g., DerivedData/sub-114/ses-Baseline -> sub-114)
        # derived_dir is like .../DerivedData/sub-XXX/ses-YYY
        subject_id = derived_dir.parent.name if derived_dir.parent.name.startswith("sub-") else None

        # Detect the two central FOV vertebrae
        detection_result = detect_central_vertebrae(seg_dir, subject_id=subject_id)

        # Log QC messages
        for msg in detection_result.qc_messages:
            logger.info(msg)

        if not detection_result.success:
            results["errors"].append("Vertebra detection failed - could not find 2 valid vertebrae")
            return results

        # Save detection result for QC
        save_detection_result(detection_result, vb_dir / "vertebra_detection.json")

        # Create standardized L1/L2 body files with vertebral body isolation
        vertebra_paths = standardize_vertebrae(
            detection_result, vb_dir,
            isolate_body_func=isolate_vertebral_body
        )

        results["vertebral_bodies_dir"] = vb_dir
        results["vertebra_detection"] = {
            "l1_original": detection_result.l1_info.original_label,
            "l2_original": detection_result.l2_info.original_label,
            "l1_voxel_count": detection_result.l1_info.voxel_count,
            "l2_voxel_count": detection_result.l2_info.voxel_count,
        }

        # Load the standardized bodies for downstream use
        l1_body = nib.load(vertebra_paths["L1"])
        l2_body = nib.load(vertebra_paths["L2"])

        # Get z-ranges
        l1_z = np.where(l1_body.get_fdata().astype(bool).any(axis=(0, 1)))[0]
        l2_z = np.where(l2_body.get_fdata().astype(bool).any(axis=(0, 1)))[0]
        results["l1_z_range"] = (int(l1_z.min()), int(l1_z.max())) if len(l1_z) > 0 else None
        results["l2_z_range"] = (int(l2_z.min()), int(l2_z.max())) if len(l2_z) > 0 else None

        logger.info(f"Vertebral bodies isolated: L1 at z={results['l1_z_range']}, "
                    f"L2 at z={results['l2_z_range']}")

    except Exception as e:
        results["errors"].append(f"Vertebral body detection/isolation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return results

    # Step 3: Create muscle compartment envelope
    try:
        # Try multiple naming conventions (TotalSegmentator varies by version)
        muscle_names = [
            ("erector_spinae_left.nii.gz", "erector_spinae_right.nii.gz"),
            ("autochthon_left.nii.gz", "autochthon_right.nii.gz"),
        ]

        left_path = None
        right_path = None
        for left_name, right_name in muscle_names:
            l_path = seg_dir / left_name
            r_path = seg_dir / right_name
            if l_path.exists() and r_path.exists():
                left_path = l_path
                right_path = r_path
                logger.info(f"Using muscle masks: {left_name}, {right_name}")
                break

        if left_path is None or right_path is None:
            results["errors"].append("Erector spinae/autochthon masks not found")
            return results

        left_nii = nib.load(left_path)
        right_nii = nib.load(right_path)

        envelope = create_muscle_envelope(left_nii, right_nii, l1_body, l2_body)

        envelope_path = derived_dir / "muscle_compartment.nii.gz"
        nib.save(envelope, envelope_path)

        results["muscle_envelope_path"] = envelope_path
        logger.info(f"Muscle compartment envelope saved: {envelope_path}")

    except Exception as e:
        results["errors"].append(f"Muscle envelope creation failed: {e}")
        return results

    results["success"] = True
    return results


if __name__ == "__main__":
    # Test with a single subject
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python segmentation.py <ct_path> <output_dir>")
        sys.exit(1)

    ct_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    results = process_segmentation(ct_path, output_dir)
    print(f"Results: {results}")
