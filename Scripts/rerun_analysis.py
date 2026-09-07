#!/usr/bin/env python3
"""
rerun_analysis.py - Re-run the analysis steps without TotalSegmentator

Uses the existing TotalSegmentator outputs (roi_subset, tissue_4_types and
vertebrae_body tasks) and the existing manual phantom calibration, and re-runs:

    1. vertebra selection (two largest complete vertebrae; subject overrides)
       and body-only isolation from the vertebrae_body mask
    2. muscle compartment envelope
    3. bone analysis (co-located per-vertebra calibration)
    4. muscle analysis (+ tissue masks)
    5. adipose analysis (not reported at group level)
    6. IMAT validation against TotalSegmentator tissue_4_types
    7. QC images
    8. analysis_manifest.json (provenance)

Prerequisites per session (all produced once, never overwritten here):
    segmentations/                      TotalSegmentator roi_subset + tissue_4_types
    segmentations/vertebrae_body/       01b_segment_vertebral_bodies.py
    phantom_calibration.json etc.       manual_calibration.py
    phantom_zprofile.json               02_phantom_zprofile.py

Usage:
    python Scripts/rerun_analysis.py --all
    python Scripts/rerun_analysis.py --subject sub-101
    python Scripts/rerun_analysis.py --subject sub-101 --session ses-Baseline

This OVERWRITES bone/muscle/adipose/validation results, vertebral_bodies/,
muscle_compartment.nii.gz, muscle_masks/ and the QC images of the session.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

segmentation = import_module("01_segmentation")
bone_analysis = import_module("03_bone_analysis")
muscle_analysis = import_module("04_muscle_analysis")
adipose_analysis = import_module("05_adipose_analysis")
tissue_validation = import_module("07_tissue_validation")
from utils.qc_visualization import generate_all_qc_images  # noqa: E402
from utils.vertebra_detection import detect_central_vertebrae, standardize_vertebrae, save_detection_result  # noqa: E402
from utils.provenance import provenance  # noqa: E402

PROJECT_ROOT = SCRIPTS_DIR.parent
RAW_DATA = PROJECT_ROOT.parent / "RawData" / "bmd_ct"
DERIVED_DATA = PROJECT_ROOT / "DerivedData"
QC_DIR = PROJECT_ROOT / "QC"
SESSIONS = ["ses-Baseline", "ses-Followup"]


def find_ct_path(subject_id: str, session: str) -> Path:
    """The single 1.25 mm CT of a session (ambiguity is an error, review L08)."""
    session_ct_dir = RAW_DATA / subject_id / session / "ct"
    ct_files = sorted(session_ct_dir.glob("*_rec-stnd1.25mm_ct.nii.gz"))
    if len(ct_files) != 1:
        raise FileNotFoundError(f"{subject_id}/{session}: expected exactly one 1.25 mm CT, found {len(ct_files)}")
    return ct_files[0]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict:
    out = {}
    for name in ["numpy", "scipy", "nibabel", "scikit-image", "pandas", "TotalSegmentator"]:
        try:
            from importlib.metadata import version
            out[name] = version(name)
        except Exception:
            out[name] = "unknown"
    return out


def regenerate_vertebral_bodies(seg_dir: Path, vb_dir: Path, subject_id: str):
    """Vertebra selection (with subject overrides) + body-only isolation."""
    import nibabel as nib
    vb_dir.mkdir(parents=True, exist_ok=True)

    detection_result = detect_central_vertebrae(seg_dir, subject_id=subject_id)
    for msg in detection_result.qc_messages:
        logger.info(msg)
    if not detection_result.success:
        logger.error("Vertebra detection failed")
        return None, None
    save_detection_result(detection_result, vb_dir / "vertebra_detection.json")

    body_mask_path = seg_dir / "vertebrae_body" / "vertebrae_body.nii.gz"
    if not body_mask_path.exists():
        raise FileNotFoundError(
            f"{body_mask_path} missing. Run Scripts/01b_segment_vertebral_bodies.py first "
            "(body-only isolation is required; the legacy largest-component isolation is not used).")
    vertebra_paths = standardize_vertebrae(detection_result, vb_dir, body_mask_path=body_mask_path)
    return nib.load(vertebra_paths["L1"]), nib.load(vertebra_paths["L2"])


def regenerate_muscle_envelope(seg_dir: Path, derived_dir: Path, l1_nii, l2_nii) -> Path:
    import nibabel as nib
    left_path, right_path = seg_dir / "autochthon_left.nii.gz", seg_dir / "autochthon_right.nii.gz"
    if not left_path.exists():
        left_path, right_path = seg_dir / "erector_spinae_left.nii.gz", seg_dir / "erector_spinae_right.nii.gz"
    if not (left_path.exists() and right_path.exists()):
        raise FileNotFoundError("Erector spinae / autochthon masks not found")
    envelope = segmentation.create_muscle_envelope(nib.load(left_path), nib.load(right_path), l1_nii, l2_nii)
    envelope_path = derived_dir / "muscle_compartment.nii.gz"
    nib.save(envelope, envelope_path)
    logger.info(f"Saved {envelope_path}")
    return envelope_path


def process_session(subject_id: str, session: str, skip_sat: bool = False) -> bool:
    logger.info(f"\n{'=' * 60}\nProcessing {subject_id} / {session}\n{'=' * 60}")
    derived_dir = DERIVED_DATA / subject_id / session
    qc_dir = QC_DIR / subject_id / session
    seg_dir = derived_dir / "segmentations"
    vb_dir = derived_dir / "vertebral_bodies"
    prov = provenance()
    manifest = {"subject": subject_id, "session": session, "started": datetime.now().isoformat(),
                "script": Path(__file__).name, "git_commit": prov["git_commit"],
                "git_dirty_scripts": prov["git_dirty_scripts"], "scripts_tree_sha256": prov["scripts_tree_sha256"],
                "provenance_note": prov["note"], "packages": package_versions(), "steps": {}}

    try:
        ct_path = find_ct_path(subject_id, session)
    except FileNotFoundError as e:
        logger.error(str(e))
        return False
    if not seg_dir.exists():
        logger.error(f"Segmentations not found: {seg_dir}")
        return False
    if not (derived_dir / "phantom_zprofile.json").exists():
        logger.error(f"{derived_dir / 'phantom_zprofile.json'} missing; run Scripts/02_phantom_zprofile.py first")
        return False
    derived_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    manifest["ct"] = str(ct_path)

    # 1. vertebrae
    logger.info("\n--- Step 1: Vertebra selection & body-only isolation ---")
    l1_nii, l2_nii = regenerate_vertebral_bodies(seg_dir, vb_dir, subject_id)
    if l1_nii is None:
        return False
    manifest["steps"]["vertebral_bodies"] = json.load(open(vb_dir / "body_isolation.json"))["method"]

    # 2. envelope
    logger.info("\n--- Step 2: Muscle envelope ---")
    envelope_path = regenerate_muscle_envelope(seg_dir, derived_dir, l1_nii, l2_nii)

    # 3. bone
    logger.info("\n--- Step 3: Bone analysis ---")
    bone_result = bone_analysis.analyze_bone(ct_path, vb_dir, derived_dir)
    bone_analysis.save_bone_results(bone_result, derived_dir / "bone_results.json")
    manifest["steps"]["bone"] = {"success": bone_result.success, "calibration": bone_result.calibration_method,
                                 "erosion_mm": bone_result.erosion_mm}
    if bone_result.L1L2_vBMD_mean_mgcm3 is not None:
        logger.info(f"  L1 {bone_result.L1.vBMD_mean_mgcm3:.1f}  L2 {bone_result.L2.vBMD_mean_mgcm3:.1f}  "
                    f"L1L2 {bone_result.L1L2_vBMD_mean_mgcm3:.1f} mg/cm³")

    # 4. muscle
    logger.info("\n--- Step 4: Muscle analysis ---")
    muscle_result = muscle_analysis.analyze_muscle(ct_path, envelope_path, seg_dir, derived_dir)
    muscle_analysis.save_muscle_results(muscle_result, derived_dir / "muscle_results.json")
    muscle_analysis.save_tissue_masks(ct_path, envelope_path, derived_dir, derived_dir / "muscle_masks")
    manifest["steps"]["muscle"] = {"success": muscle_result.success, "drift": muscle_result.drift_correction_method,
                                   "classes": muscle_result.classification_convention}
    logger.info(f"  SMD {muscle_result.muscle_SMD_mean_hu:.1f} HU, low-density {muscle_result.muscle_low_density_percent:.1f}%, "
                f"IMAT {muscle_result.imat_percent:.1f}%, volume {muscle_result.muscle_tissue_volume_cm3:.1f} cm³")

    # 5. adipose (kept current; not reported at group level)
    logger.info("\n--- Step 5: Adipose analysis ---")
    try:
        adipose_result = adipose_analysis.analyze_adipose(ct_path, seg_dir, vb_dir,
                                                          muscle_csa_cm2=muscle_result.muscle_CSA_mean_cm2,
                                                          skip_sat=skip_sat)
        adipose_analysis.save_adipose_results(adipose_result, derived_dir / "adipose_results.json")
        manifest["steps"]["adipose"] = {"success": adipose_result.success, "sat_fov_adequate": adipose_result.sat_fov_adequate}
    except Exception as e:
        logger.error(f"Adipose analysis failed: {e}")
        manifest["steps"]["adipose"] = {"success": False, "error": str(e)}

    # 6. IMAT validation
    logger.info("\n--- Step 6: IMAT validation (tissue_4_types) ---")
    try:
        validation = tissue_validation.validate_tissue_segmentation(derived_dir, subject_id, session)
        tissue_validation.save_validation_results(validation, derived_dir / "validation_results.json")
        try:
            tissue_validation.create_tissue_validation_figure(ct_path, derived_dir, qc_dir / "tissue_validation.png", validation)
        except Exception as e:
            logger.warning(f"validation figure failed: {e}")
        manifest["steps"]["validation"] = {"success": validation.success, "imat_dice": validation.imat_dice}
        logger.info(f"  IMAT Dice {validation.imat_dice:.3f}")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        manifest["steps"]["validation"] = {"success": False, "error": str(e)}

    # 7. QC images
    logger.info("\n--- Step 7: QC images ---")
    try:
        generate_all_qc_images(ct_path, derived_dir, qc_dir)
    except Exception as e:
        logger.error(f"QC visualization failed: {e}")

    manifest["finished"] = datetime.now().isoformat()
    manifest["success"] = bool(bone_result.success and muscle_result.success)
    with open(derived_dir / "analysis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"\n{subject_id}/{session}: {'OK' if manifest['success'] else 'FAILED'}")
    return manifest["success"]


def main():
    parser = argparse.ArgumentParser(description="Re-run analysis without TotalSegmentator")
    parser.add_argument("--subject", help="Subject ID (e.g., sub-101)")
    parser.add_argument("--session", help="Session (e.g., ses-Baseline). Default: both.")
    parser.add_argument("--all", action="store_true", help="All subjects in DerivedData")
    parser.add_argument("--skip-sat", action="store_true", help="Skip SAT analysis")
    args = parser.parse_args()
    if not args.subject and not args.all:
        parser.error("give --subject or --all")

    subjects = [args.subject] if args.subject else sorted(p.name for p in DERIVED_DATA.glob("sub-*"))
    sessions = [args.session] if args.session else SESSIONS
    failures = []
    for sub in subjects:
        for ses in sessions:
            if not (DERIVED_DATA / sub / ses).exists():
                continue
            try:
                ok = process_session(sub, ses, args.skip_sat)
            except Exception as e:
                logger.exception(f"{sub}/{ses}: {e}")
                ok = False
            if not ok:
                failures.append(f"{sub}/{ses}")
    logger.info(f"\nDone. {len(failures)} failure(s): {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
