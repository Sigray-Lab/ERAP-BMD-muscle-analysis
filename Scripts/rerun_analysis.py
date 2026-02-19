#!/usr/bin/env python3
"""
rerun_analysis.py - Re-run analysis steps without TotalSegmentator

This script re-runs the analysis steps (bone, muscle, adipose, QC)
using existing TotalSegmentator outputs. It avoids re-running heavy
segmentation when only the analysis parameters have changed.

Usage:
    python rerun_analysis.py --subject sub-101 --session ses-Baseline
    python rerun_analysis.py --subject sub-101  # Both sessions
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add Scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dynamic imports
from importlib import import_module

segmentation = import_module("01_segmentation")
bone_analysis = import_module("03_bone_analysis")
muscle_analysis = import_module("04_muscle_analysis")
adipose_analysis = import_module("05_adipose_analysis")
from utils.qc_visualization import generate_all_qc_images
from utils.vertebra_detection import detect_central_vertebrae, standardize_vertebrae, save_detection_result


# Project paths
PROJECT_ROOT = SCRIPTS_DIR.parent
RAW_DATA = PROJECT_ROOT.parent / "RawData" / "bmd_ct"
DERIVED_DATA = PROJECT_ROOT / "DerivedData"
QC_DIR = PROJECT_ROOT / "QC"


def find_ct_path(subject_id: str, session: str) -> Path:
    """Find the 1.25mm CT for a subject/session."""
    session_ct_dir = RAW_DATA / subject_id / session / "ct"

    ct_files = list(session_ct_dir.glob("*_rec-stnd1.25mm_ct.nii.gz"))
    if ct_files:
        return ct_files[0]

    raise FileNotFoundError(f"No 1.25mm CT found in {session_ct_dir}")


def regenerate_vertebral_bodies(seg_dir: Path, vb_dir: Path):
    """Re-generate vertebral body masks using robust detection."""
    import nibabel as nib

    vb_dir.mkdir(parents=True, exist_ok=True)

    # Use robust vertebra detection
    detection_result = detect_central_vertebrae(seg_dir)

    # Log QC messages
    for msg in detection_result.qc_messages:
        logger.info(msg)

    if not detection_result.success:
        logger.error("Vertebra detection failed")
        return None, None

    # Save detection result for QC
    save_detection_result(detection_result, vb_dir / "vertebra_detection.json")

    # Create standardized L1/L2 body files with vertebral body isolation
    vertebra_paths = standardize_vertebrae(
        detection_result, vb_dir,
        isolate_body_func=segmentation.isolate_vertebral_body
    )

    # Load and return for downstream use
    l1_nii = nib.load(vertebra_paths["L1"]) if "L1" in vertebra_paths else None
    l2_nii = nib.load(vertebra_paths["L2"]) if "L2" in vertebra_paths else None

    return l1_nii, l2_nii


def regenerate_muscle_envelope(seg_dir: Path, vb_dir: Path, derived_dir: Path, l1_nii=None, l2_nii=None):
    """Re-generate muscle compartment envelope."""
    import nibabel as nib

    # Try autochthon naming first (TotalSegmentator varies)
    left_path = seg_dir / "autochthon_left.nii.gz"
    right_path = seg_dir / "autochthon_right.nii.gz"

    if not left_path.exists():
        left_path = seg_dir / "erector_spinae_left.nii.gz"
        right_path = seg_dir / "erector_spinae_right.nii.gz"

    if not left_path.exists() or not right_path.exists():
        logger.warning("Erector spinae masks not found")
        return

    # Load vertebral body masks if not provided
    if l1_nii is None or l2_nii is None:
        l1_path = vb_dir / "L1_body.nii.gz"
        l2_path = vb_dir / "L2_body.nii.gz"

        if not l1_path.exists() or not l2_path.exists():
            logger.warning("L1/L2 body masks not found, cannot create envelope")
            return

        l1_nii = nib.load(l1_path)
        l2_nii = nib.load(l2_path)

    logger.info("Creating muscle compartment envelope...")
    left_nii = nib.load(left_path)
    right_nii = nib.load(right_path)

    envelope = segmentation.create_muscle_envelope(left_nii, right_nii, l1_nii, l2_nii)

    envelope_path = derived_dir / "muscle_compartment.nii.gz"
    nib.save(envelope, envelope_path)
    logger.info(f"Saved {envelope_path}")


def process_session(subject_id: str, session: str, skip_sat: bool = False):
    """Re-run analysis for a single session."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {subject_id} / {session}")
    logger.info(f"{'='*60}")

    # Paths
    derived_dir = DERIVED_DATA / subject_id / session
    qc_dir = QC_DIR / subject_id / session
    seg_dir = derived_dir / "segmentations"
    vb_dir = derived_dir / "vertebral_bodies"

    # Find CT
    try:
        ct_path = find_ct_path(subject_id, session)
        logger.info(f"CT: {ct_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return False

    # Check for existing segmentations
    if not seg_dir.exists():
        logger.error(f"Segmentations not found: {seg_dir}")
        logger.error("Run full pipeline first or run TotalSegmentator manually")
        return False

    # Make output dirs
    derived_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Robust vertebra detection and body isolation (L1/L2)
    logger.info("\n--- Step 1: Vertebra Detection & Body Isolation ---")
    l1_nii, l2_nii = regenerate_vertebral_bodies(seg_dir, vb_dir)

    if l1_nii is None or l2_nii is None:
        logger.error("Failed to detect vertebrae")
        return False

    # Step 2: Re-generate muscle envelope
    logger.info("\n--- Step 2: Muscle Envelope ---")
    regenerate_muscle_envelope(seg_dir, vb_dir, derived_dir, l1_nii, l2_nii)

    # Step 3: Bone Analysis
    logger.info("\n--- Step 3: Bone Analysis ---")
    try:
        bone_result = bone_analysis.analyze_bone(ct_path, vb_dir, derived_dir)
        bone_analysis.save_bone_results(bone_result, derived_dir / "bone_results.json")

        if bone_result.L1:
            logger.info(f"  L1 vBMD: {bone_result.L1.vBMD_mean_mgcm3:.1f} mg/cm³")
        if bone_result.L2:
            logger.info(f"  L2 vBMD: {bone_result.L2.vBMD_mean_mgcm3:.1f} mg/cm³")
        if bone_result.L1L2_vBMD_mean_mgcm3:
            logger.info(f"  L1L2 vBMD: {bone_result.L1L2_vBMD_mean_mgcm3:.1f} mg/cm³")
    except Exception as e:
        logger.error(f"Bone analysis failed: {e}")

    # Step 4: Muscle Analysis
    logger.info("\n--- Step 4: Muscle Analysis ---")
    envelope_path = derived_dir / "muscle_compartment.nii.gz"
    try:
        if envelope_path.exists():
            muscle_result = muscle_analysis.analyze_muscle(ct_path, envelope_path, seg_dir, derived_dir)
            muscle_analysis.save_muscle_results(muscle_result, derived_dir / "muscle_results.json")

            # Save tissue masks
            muscle_masks_dir = derived_dir / "muscle_masks"
            muscle_analysis.save_tissue_masks(ct_path, envelope_path, derived_dir, muscle_masks_dir)

            logger.info(f"  SMD: {muscle_result.muscle_SMD_mean_hu:.1f} HU")
            logger.info(f"  Low density: {muscle_result.muscle_low_density_percent:.1f}%")
            logger.info(f"  IMAT: {muscle_result.imat_percent:.1f}%")
            logger.info(f"  Saved masks to: {muscle_masks_dir}")
        else:
            logger.warning("Muscle envelope not found")
    except Exception as e:
        logger.error(f"Muscle analysis failed: {e}")

    # Step 5: Adipose Analysis
    logger.info("\n--- Step 5: Adipose Analysis ---")
    try:
        # Get muscle CSA for ratio
        muscle_csa = None
        muscle_results_path = derived_dir / "muscle_results.json"
        if muscle_results_path.exists():
            with open(muscle_results_path) as f:
                muscle_data = json.load(f)
            muscle_csa = muscle_data.get("muscle_CSA_mean_cm2")

        adipose_result = adipose_analysis.analyze_adipose(
            ct_path, seg_dir, vb_dir,
            muscle_csa_cm2=muscle_csa,
            skip_sat=skip_sat
        )
        adipose_analysis.save_adipose_results(adipose_result, derived_dir / "adipose_results.json")

        logger.info(f"  VAT volume: {adipose_result.vat_volume_cm3:.1f} cm³")
        if adipose_result.sat_volume_cm3:
            logger.info(f"  SAT volume: {adipose_result.sat_volume_cm3:.1f} cm³")
    except Exception as e:
        logger.error(f"Adipose analysis failed: {e}")

    # Step 6: QC Visualization
    logger.info("\n--- Step 6: QC Visualization ---")
    try:
        generate_all_qc_images(ct_path, derived_dir, qc_dir)
        logger.info(f"  QC images saved to: {qc_dir}")
    except Exception as e:
        logger.error(f"QC visualization failed: {e}")

    logger.info(f"\nProcessing complete for {subject_id} / {session}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Re-run analysis without TotalSegmentator")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-101)")
    parser.add_argument("--session", help="Session (e.g., ses-Baseline). If not specified, runs both.")
    parser.add_argument("--skip-sat", action="store_true", help="Skip SAT analysis")

    args = parser.parse_args()

    sessions = [args.session] if args.session else ["ses-Baseline", "ses-Followup"]

    success = True
    for session in sessions:
        if not process_session(args.subject, session, args.skip_sat):
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
