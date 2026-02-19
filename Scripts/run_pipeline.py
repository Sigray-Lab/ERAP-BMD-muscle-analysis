#!/usr/bin/env python3
"""
run_pipeline.py - Main orchestrator for ERAP CT analysis pipeline

This script coordinates all processing steps:
1. Segmentation (TotalSegmentator, vertebral body isolation, muscle envelope)
2. Phantom calibration
3. Bone analysis (trabecular BMD)
4. Muscle analysis (SMD, IMAT)
5. Adipose analysis (VAT, conditional SAT)
6. QC visualization
7. Results aggregation

Usage:
    python run_pipeline.py --data <raw_data_dir> --output <output_dir>
    python run_pipeline.py --data <raw_data_dir> --output <output_dir> --subject sub-101
    python run_pipeline.py --data <raw_data_dir> --output <output_dir> --skip-sat
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

# Add Scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Import pipeline modules (now with numbered prefixes)
from importlib import import_module

# Dynamic imports to handle numbered prefixes
segmentation = import_module("01_segmentation")
phantom_calibration = import_module("02_phantom_calibration")
bone_analysis = import_module("03_bone_analysis")
muscle_analysis = import_module("04_muscle_analysis")
adipose_analysis = import_module("05_adipose_analysis")
results_aggregation = import_module("06_results_aggregation")
tissue_validation = import_module("07_tissue_validation")

# Import from utils
from utils.logger import PipelineLogger, setup_pipeline_logging, set_logger
from utils.qc_visualization import generate_all_qc_images


def find_ct_scans(data_dir: Path, subject_filter: Optional[str] = None) -> List[Tuple[str, str, Path]]:
    """
    Find all CT scans to process.

    Args:
        data_dir: Root directory containing subject folders
        subject_filter: Optional subject ID to process only one subject

    Returns:
        List of (subject_id, session, ct_path) tuples
    """
    scans = []

    # Find subject directories
    subject_dirs = sorted([d for d in data_dir.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        # Apply filter if specified
        if subject_filter and subject_id != subject_filter:
            continue

        # Look for sessions
        for session in ["ses-Baseline", "ses-Followup"]:
            session_dir = subject_dir / session / "ct"

            if not session_dir.exists():
                continue

            # Find 1.25mm CT (preferred resolution)
            ct_files = list(session_dir.glob("*_rec-stnd1.25mm_ct.nii.gz"))

            if ct_files:
                ct_path = ct_files[0]
                scans.append((subject_id, session, ct_path))

    return scans


def process_single_scan(subject_id: str,
                        session: str,
                        ct_path: Path,
                        output_base: Path,
                        skip_sat: bool = False,
                        force: bool = False,
                        logger: PipelineLogger = None) -> dict:
    """
    Process a single CT scan through the full pipeline.

    Args:
        subject_id: Subject identifier (e.g., "sub-101")
        session: Session name (e.g., "ses-Baseline")
        ct_path: Path to CT NIfTI file
        output_base: Base output directory
        skip_sat: Whether to skip SAT analysis
        logger: PipelineLogger instance

    Returns:
        Dictionary with processing results and status
    """
    results = {
        "subject_id": subject_id,
        "session": session,
        "ct_path": str(ct_path),
        "success": False,
        "errors": []
    }

    # Set up output directories
    derived_dir = output_base / "DerivedData" / subject_id / session
    qc_dir = output_base / "QC" / subject_id / session

    derived_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Segmentation
    logger.section("Step 1: Segmentation")
    try:
        seg_results = segmentation.process_segmentation(ct_path, derived_dir, force=force)
        if not seg_results["success"]:
            results["errors"].append(f"Segmentation failed: {seg_results['errors']}")
            logger.error(f"Segmentation failed: {seg_results['errors']}")
            return results
        results["segmentation"] = seg_results
        logger.log("Segmentation completed successfully", indent=1)
    except Exception as e:
        results["errors"].append(f"Segmentation error: {e}")
        logger.error(f"Segmentation error: {e}")
        return results

    # Step 2: Phantom Calibration
    logger.section("Step 2: Phantom Calibration")
    try:
        cal_result = phantom_calibration.calibrate_phantom(ct_path, derived_dir)
        if not cal_result.success:
            results["errors"].append("Phantom calibration failed")
            logger.warn("Phantom calibration failed")
            # Continue anyway - may have partial results
        results["calibration"] = {
            "success": cal_result.success,
            "r_squared": cal_result.r_squared,
            "slope": cal_result.slope,
            "intercept": cal_result.intercept,
            "qc_passed": cal_result.qc_passed
        }
        logger.metric("R²", f"{cal_result.r_squared:.4f}", indent=1)
        logger.metric("Slope", f"{cal_result.slope:.4f}", indent=1)
        logger.metric("Intercept", f"{cal_result.intercept:.2f}", "mg/cm³", indent=1)
    except Exception as e:
        results["errors"].append(f"Calibration error: {e}")
        logger.error(f"Calibration error: {e}")
        return results

    # Step 3: Bone Analysis
    logger.section("Step 3: Bone Analysis")
    try:
        vb_dir = derived_dir / "vertebral_bodies"
        bone_result = bone_analysis.analyze_bone(ct_path, vb_dir, derived_dir)

        bone_output = derived_dir / "bone_results.json"
        bone_analysis.save_bone_results(bone_result, bone_output)

        results["bone"] = {
            "success": bone_result.success,
            "L1L2_vBMD_mean": bone_result.L1L2_vBMD_mean_mgcm3
        }

        if bone_result.L1L2_vBMD_mean_mgcm3:
            if bone_result.L1:
                logger.metric("L1 vBMD", f"{bone_result.L1.vBMD_mean_mgcm3:.1f}", "mg/cm³", indent=1)
            if bone_result.L2:
                logger.metric("L2 vBMD", f"{bone_result.L2.vBMD_mean_mgcm3:.1f}", "mg/cm³", indent=1)
            logger.metric("L1L2 vBMD (mean)", f"{bone_result.L1L2_vBMD_mean_mgcm3:.1f}", "mg/cm³", indent=1)
        else:
            logger.warn("Could not calculate L1L2 vBMD")

    except Exception as e:
        results["errors"].append(f"Bone analysis error: {e}")
        logger.error(f"Bone analysis error: {e}")

    # Step 4: Muscle Analysis
    logger.section("Step 4: Muscle Analysis")
    try:
        envelope_path = derived_dir / "muscle_compartment.nii.gz"
        seg_dir = derived_dir / "segmentations"

        if envelope_path.exists():
            muscle_result = muscle_analysis.analyze_muscle(ct_path, envelope_path, seg_dir, derived_dir)

            muscle_output = derived_dir / "muscle_results.json"
            muscle_analysis.save_muscle_results(muscle_result, muscle_output)

            # Save muscle tissue classification masks
            muscle_masks_dir = derived_dir / "muscle_masks"
            muscle_analysis.save_tissue_masks(ct_path, envelope_path, derived_dir, muscle_masks_dir)
            logger.log(f"Muscle masks saved to: {muscle_masks_dir}", indent=1)

            results["muscle"] = {
                "success": muscle_result.success,
                "SMD_mean": muscle_result.muscle_SMD_mean_hu,
                "IMAT_percent": muscle_result.imat_percent,
                "low_density_percent": muscle_result.muscle_low_density_percent
            }

            logger.metric("SMD (mean)", f"{muscle_result.muscle_SMD_mean_hu:.1f}", "HU", indent=1)
            logger.metric("SMD (median)", f"{muscle_result.muscle_SMD_median_hu:.1f}", "HU", indent=1)
            logger.metric("Low density muscle", f"{muscle_result.muscle_low_density_percent:.1f}", "%", indent=1)
            logger.metric("IMAT", f"{muscle_result.imat_percent:.1f}", "%", indent=1)
            logger.metric("Muscle volume", f"{muscle_result.muscle_tissue_volume_cm3:.1f}", "cm³", indent=1)
        else:
            results["errors"].append("Muscle envelope not found")
            logger.warn("Muscle envelope not found, skipping muscle analysis")

    except Exception as e:
        results["errors"].append(f"Muscle analysis error: {e}")
        logger.error(f"Muscle analysis error: {e}")

    # Step 5: Adipose Analysis
    logger.section("Step 5: Adipose Analysis")
    try:
        seg_dir = derived_dir / "segmentations"
        vb_dir = derived_dir / "vertebral_bodies"

        # Get muscle CSA for ratio calculation
        muscle_csa = None
        if "muscle" in results and results["muscle"].get("success"):
            muscle_data = json.loads((derived_dir / "muscle_results.json").read_text())
            muscle_csa = muscle_data.get("muscle_CSA_mean_cm2")

        adipose_result = adipose_analysis.analyze_adipose(
            ct_path, seg_dir, vb_dir,
            muscle_csa_cm2=muscle_csa,
            skip_sat=skip_sat
        )

        adipose_output = derived_dir / "adipose_results.json"
        adipose_analysis.save_adipose_results(adipose_result, adipose_output)

        results["adipose"] = {
            "success": adipose_result.success,
            "VAT_volume": adipose_result.vat_volume_cm3,
            "SAT_fov_adequate": adipose_result.sat_fov_adequate,
            "SAT_volume": adipose_result.sat_volume_cm3
        }

        logger.metric("VAT volume", f"{adipose_result.vat_volume_cm3:.1f}", "cm³", indent=1)
        if adipose_result.sat_volume_cm3:
            logger.metric("SAT volume", f"{adipose_result.sat_volume_cm3:.1f}", "cm³ (FOV adequate)", indent=1)
        elif not adipose_result.sat_fov_adequate:
            logger.info("SAT excluded (FOV inadequate)", indent=1)

    except Exception as e:
        results["errors"].append(f"Adipose analysis error: {e}")
        logger.error(f"Adipose analysis error: {e}")

    # Step 6: QC Visualization
    logger.section("Step 6: QC Visualization")
    try:
        generate_all_qc_images(ct_path, derived_dir, qc_dir)
        results["qc_images"] = str(qc_dir)
        logger.log(f"QC images saved to: {qc_dir}", indent=1)
    except Exception as e:
        results["errors"].append(f"QC visualization error: {e}")
        logger.error(f"QC visualization error: {e}")

    # Step 7: Tissue Validation (compare HU-based vs TotalSegmentator tissue_4_types)
    logger.section("Step 7: Tissue Validation")
    try:
        validation_result = tissue_validation.validate_tissue_segmentation(
            derived_dir, subject_id, session
        )

        # Save validation results
        validation_output = derived_dir / "validation_results.json"
        tissue_validation.save_validation_results(validation_result, validation_output)

        # Generate validation QC figure
        validation_qc_path = qc_dir / "tissue_validation.png"
        try:
            tissue_validation.create_tissue_validation_figure(
                ct_path, derived_dir, validation_qc_path, validation_result
            )
            logger.info(f"Validation QC image saved: {validation_qc_path}", indent=1)
        except Exception as qc_error:
            logger.warn(f"Could not generate validation QC image: {qc_error}", indent=1)

        results["validation"] = {
            "success": validation_result.success,
            "imat_dice": validation_result.imat_dice if not np.isnan(validation_result.imat_dice) else None,
            "imat_hu_volume_cm3": validation_result.imat_hu_volume_cm3,
            "imat_ts_in_envelope_volume_cm3": validation_result.imat_ts_in_envelope_volume_cm3,
            "ts_imat_raw_volume_cm3": validation_result.ts_imat_raw_volume_cm3,
            "muscle_envelope_volume_cm3": validation_result.muscle_envelope_volume_cm3,
            "envelope_coverage_ok": validation_result.envelope_coverage_ok
        }

        if validation_result.success:
            if not np.isnan(validation_result.imat_dice):
                logger.metric("IMAT Dice (in envelope)", f"{validation_result.imat_dice:.3f}", indent=1)
            logger.metric("IMAT HU-based", f"{validation_result.imat_hu_volume_cm3:.1f}", "cm³", indent=1)
            logger.metric("IMAT TS (in envelope)", f"{validation_result.imat_ts_in_envelope_volume_cm3:.1f}", "cm³", indent=1)
            logger.metric("IMAT TS raw (L1-L2)", f"{validation_result.ts_imat_raw_volume_cm3:.1f}", "cm³", indent=1)

            if not validation_result.envelope_coverage_ok:
                logger.warn("Low envelope coverage: TS IMAT may include phantom artifacts", indent=1)

            for msg in validation_result.qc_messages:
                if msg.startswith("WARNING"):
                    logger.warn(msg, indent=1)
        else:
            logger.warn("Tissue validation incomplete (tissue_4_types may not have run)")

    except Exception as e:
        results["errors"].append(f"Tissue validation error: {e}")
        logger.error(f"Tissue validation error: {e}")

    # Determine overall success
    results["success"] = (
        results.get("bone", {}).get("success", False) or
        results.get("muscle", {}).get("success", False) or
        results.get("adipose", {}).get("success", False)
    )

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ERAP CT Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process all subjects:
    python run_pipeline.py --data ../RawData/bmd_ct --output .

  Process single subject:
    python run_pipeline.py --data ../RawData/bmd_ct --output . --subject sub-101

  Skip SAT analysis:
    python run_pipeline.py --data ../RawData/bmd_ct --output . --skip-sat
        """
    )

    parser.add_argument("--data", required=True, type=Path,
                        help="Path to raw data directory containing subject folders")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to output directory")
    parser.add_argument("--subject", type=str, default=None,
                        help="Process only this subject (e.g., sub-101)")
    parser.add_argument("--skip-sat", action="store_true",
                        help="Skip subcutaneous adipose tissue analysis")
    parser.add_argument("--skip-aggregation", action="store_true",
                        help="Skip final results aggregation")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run of TotalSegmentator even if output exists")

    args = parser.parse_args()

    # Validate paths
    if not args.data.exists():
        print(f"Error: Data directory not found: {args.data}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # Set up logging (ONH-style format)
    logger = setup_pipeline_logging(args.output, prefix="pipeline")
    set_logger(logger)

    logger.blank()
    logger.log(f"Data directory: {args.data}")
    logger.log(f"Output directory: {args.output}")
    logger.log(f"Subject filter: {args.subject or 'None (all subjects)'}")
    logger.log(f"Skip SAT: {args.skip_sat}")
    logger.log(f"Force TotalSegmentator: {args.force}")

    # Find CT scans to process
    scans = find_ct_scans(args.data, args.subject)

    if not scans:
        logger.error("No CT scans found to process")
        sys.exit(1)

    logger.blank()
    logger.log(f"Found {len(scans)} CT scans to process:")
    for subject_id, session, ct_path in scans:
        logger.log(f"  {subject_id} / {session}", indent=1)

    # Process each scan
    all_results = []
    success_count = 0
    fail_count = 0

    for i, (subject_id, session, ct_path) in enumerate(scans, 1):
        logger.blank()
        logger.header(f"[{i}/{len(scans)}] {subject_id} / {session}")
        logger.log(f"CT: {ct_path}")

        try:
            result = process_single_scan(
                subject_id, session, ct_path,
                args.output,
                skip_sat=args.skip_sat,
                force=args.force,
                logger=logger
            )
            all_results.append(result)

            if result["success"]:
                success_count += 1
                logger.blank()
                logger.log("Scan processing: SUCCESS", indent=0)
            else:
                fail_count += 1
                logger.blank()
                logger.log("Scan processing: FAILED", indent=0)
                if result["errors"]:
                    for err in result["errors"]:
                        logger.warn(err, indent=1)

        except Exception as e:
            logger.error(f"Unexpected error processing {subject_id} {session}: {e}")
            fail_count += 1

    # Aggregate results
    if not args.skip_aggregation:
        logger.blank()
        logger.header("Results Aggregation")

        try:
            derived_dir = args.output / "DerivedData"
            outputs_dir = args.output / "Outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            output_csv = outputs_dir / "results.csv"
            df = results_aggregation.aggregate_results(derived_dir, output_csv)

            if len(df) > 0:
                results_aggregation.generate_summary_statistics(df, outputs_dir)

            logger.log(f"Results aggregated to: {output_csv}")

        except Exception as e:
            logger.error(f"Results aggregation failed: {e}")

    # Summary
    logger.blank()
    logger.header("Pipeline Summary")
    logger.log(f"Total scans processed: {len(scans)}")
    logger.log(f"Successful: {success_count}")
    logger.log(f"Failed: {fail_count}")

    if fail_count > 0:
        logger.blank()
        logger.log("Failed scans:")
        for result in all_results:
            if not result.get("success"):
                logger.log(f"  {result['subject_id']} {result['session']}", indent=1)
                for err in result.get("errors", []):
                    logger.warn(err, indent=2)

    logger.blank()
    logger.log(f"Pipeline complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.close()

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
