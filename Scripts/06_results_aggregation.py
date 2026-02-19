#!/usr/bin/env python3
"""
results_aggregation.py - Compile all subject results into final CSV

This module handles:
1. Loading individual subject/session JSON results
2. Computing change metrics (post - pre)
3. Compiling comprehensive results CSV
4. Generating summary statistics
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SubjectResults:
    """Results for a single subject across sessions."""
    subject_id: str
    baseline: Optional[Dict[str, Any]] = None
    followup: Optional[Dict[str, Any]] = None


def load_session_results(derived_dir: Path) -> Dict[str, Any]:
    """
    Load all analysis results for a single session.

    Args:
        derived_dir: Directory containing session-specific derived data

    Returns:
        Dictionary with all metrics from bone, muscle, adipose, and validation analyses
    """
    results = {}

    # Load bone results
    bone_path = derived_dir / "bone_results.json"
    if bone_path.exists():
        with open(bone_path) as f:
            bone = json.load(f)
            results["bone"] = bone
            results["bone_success"] = bone.get("success", False)
    else:
        results["bone_success"] = False

    # Load muscle results
    muscle_path = derived_dir / "muscle_results.json"
    if muscle_path.exists():
        with open(muscle_path) as f:
            muscle = json.load(f)
            results["muscle"] = muscle
            results["muscle_success"] = muscle.get("success", False)
    else:
        results["muscle_success"] = False

    # Load adipose results
    adipose_path = derived_dir / "adipose_results.json"
    if adipose_path.exists():
        with open(adipose_path) as f:
            adipose = json.load(f)
            results["adipose"] = adipose
            results["adipose_success"] = adipose.get("success", False)
    else:
        results["adipose_success"] = False

    # Load calibration info
    cal_path = derived_dir / "calibration_bmd.json"
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
            results["calibration_r_squared"] = cal.get("regression", {}).get("r_squared", np.nan)

    # Load validation results (tissue_4_types comparison)
    validation_path = derived_dir / "validation_results.json"
    if validation_path.exists():
        with open(validation_path) as f:
            validation = json.load(f)
            results["validation"] = validation
            results["validation_success"] = validation.get("success", False)
    else:
        results["validation_success"] = False

    return results


def flatten_results(results: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested results dictionary for CSV export.

    Args:
        results: Nested dictionary of results
        prefix: Prefix for flattened keys

    Returns:
        Flat dictionary with prefixed keys
    """
    flat = {}

    # Bone metrics
    if "bone" in results:
        bone = results["bone"]
        if bone.get("L1"):
            flat[f"{prefix}L1_vBMD_mean_mgcm3"] = bone["L1"].get("vBMD_mean_mgcm3")
            flat[f"{prefix}L1_vBMD_median_mgcm3"] = bone["L1"].get("vBMD_median_mgcm3")
            flat[f"{prefix}L1_trabecular_volume_cm3"] = bone["L1"].get("trabecular_volume_cm3")
        if bone.get("L2"):
            flat[f"{prefix}L2_vBMD_mean_mgcm3"] = bone["L2"].get("vBMD_mean_mgcm3")
            flat[f"{prefix}L2_vBMD_median_mgcm3"] = bone["L2"].get("vBMD_median_mgcm3")
            flat[f"{prefix}L2_trabecular_volume_cm3"] = bone["L2"].get("trabecular_volume_cm3")
        flat[f"{prefix}L1L2_vBMD_mean_mgcm3"] = bone.get("L1L2_vBMD_mean_mgcm3")

    # Muscle metrics
    if "muscle" in results:
        muscle = results["muscle"]
        flat[f"{prefix}muscle_compartment_volume_cm3"] = muscle.get("compartment_volume_cm3")
        flat[f"{prefix}muscle_tissue_volume_cm3"] = muscle.get("muscle_tissue_volume_cm3")
        flat[f"{prefix}muscle_SMD_mean_hu"] = muscle.get("muscle_SMD_mean_hu")
        flat[f"{prefix}muscle_SMD_median_hu"] = muscle.get("muscle_SMD_median_hu")
        flat[f"{prefix}muscle_SMD_P10_hu"] = muscle.get("muscle_SMD_P10_hu")
        flat[f"{prefix}muscle_SMD_P90_hu"] = muscle.get("muscle_SMD_P90_hu")
        flat[f"{prefix}muscle_low_density_percent"] = muscle.get("muscle_low_density_percent")
        flat[f"{prefix}IMAT_volume_cm3"] = muscle.get("imat_volume_cm3")
        flat[f"{prefix}IMAT_percent"] = muscle.get("imat_percent")
        flat[f"{prefix}muscle_CSA_mean_cm2"] = muscle.get("muscle_CSA_mean_cm2")
        flat[f"{prefix}muscle_CSA_max_cm2"] = muscle.get("muscle_CSA_max_cm2")
        flat[f"{prefix}muscle_LR_symmetry_index"] = muscle.get("muscle_LR_symmetry_index")

    # Adipose metrics
    if "adipose" in results:
        adipose = results["adipose"]
        flat[f"{prefix}VAT_volume_cm3"] = adipose.get("vat_volume_cm3")
        flat[f"{prefix}VAT_area_L1L2_cm2"] = adipose.get("vat_area_L1L2_cm2")
        flat[f"{prefix}VAT_hu_median"] = adipose.get("vat_hu_median")
        flat[f"{prefix}VAT_outlier_fraction"] = adipose.get("vat_outlier_fraction")
        flat[f"{prefix}SAT_fov_adequate"] = adipose.get("sat_fov_adequate")
        flat[f"{prefix}SAT_volume_cm3"] = adipose.get("sat_volume_cm3")
        flat[f"{prefix}VAT_SAT_ratio"] = adipose.get("vat_sat_ratio")
        flat[f"{prefix}muscle_CSA_VAT_ratio"] = adipose.get("muscle_csa_vat_ratio")

    # Validation metrics (tissue_4_types comparison - IMAT within muscle envelope)
    if "validation" in results:
        validation = results["validation"]
        flat[f"{prefix}imat_dice"] = validation.get("imat_dice")
        flat[f"{prefix}imat_hu_volume_cm3"] = validation.get("imat_hu_volume_cm3")
        # TS IMAT restricted to muscle envelope (for longitudinal comparison)
        flat[f"{prefix}imat_ts_in_envelope_cm3"] = validation.get("imat_ts_in_envelope_volume_cm3")
        flat[f"{prefix}imat_volume_ratio"] = validation.get("imat_volume_ratio")
        flat[f"{prefix}ts_imat_raw_volume_cm3"] = validation.get("ts_imat_raw_volume_cm3")
        flat[f"{prefix}muscle_envelope_volume_cm3"] = validation.get("muscle_envelope_volume_cm3")
        flat[f"{prefix}imat_dice_acceptable"] = validation.get("imat_dice_acceptable")
        flat[f"{prefix}envelope_coverage_ok"] = validation.get("envelope_coverage_ok")
        # VAT/SAT silenced due to phantom artifacts - kept as 0 for future use
        # flat[f"{prefix}ts_vat_volume_cm3"] = validation.get("ts_vat_volume_cm3")
        # flat[f"{prefix}ts_sat_volume_cm3"] = validation.get("ts_sat_volume_cm3")

    # QC metrics
    flat[f"{prefix}calibration_r_squared"] = results.get("calibration_r_squared")
    flat[f"{prefix}bone_success"] = results.get("bone_success", False)
    flat[f"{prefix}muscle_success"] = results.get("muscle_success", False)
    flat[f"{prefix}adipose_success"] = results.get("adipose_success", False)
    flat[f"{prefix}validation_success"] = results.get("validation_success", False)

    return flat


def compute_change_metrics(baseline: Dict[str, Any],
                           followup: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute change metrics (post - pre) and percent change.

    Args:
        baseline: Flattened baseline results
        followup: Flattened followup results

    Returns:
        Dictionary with change and percent change metrics
    """
    changes = {}

    # Metrics to compute changes for
    change_metrics = [
        "L1_vBMD_mean_mgcm3", "L2_vBMD_mean_mgcm3", "L1L2_vBMD_mean_mgcm3",
        "L1_trabecular_volume_cm3", "L2_trabecular_volume_cm3",
        "muscle_tissue_volume_cm3", "muscle_SMD_mean_hu",
        "muscle_low_density_percent", "IMAT_volume_cm3", "IMAT_percent",
        "muscle_CSA_mean_cm2",
        "VAT_volume_cm3", "VAT_area_L1L2_cm2",
        "SAT_volume_cm3", "VAT_SAT_ratio", "muscle_CSA_VAT_ratio",
        # Validation: TS IMAT within envelope for longitudinal comparison
        "imat_ts_in_envelope_cm3"
    ]

    for metric in change_metrics:
        pre_key = f"pre_{metric}"
        post_key = f"post_{metric}"

        pre_val = baseline.get(pre_key)
        post_val = followup.get(post_key)

        if pre_val is not None and post_val is not None:
            if not (np.isnan(pre_val) or np.isnan(post_val)):
                # Absolute change
                changes[f"{metric}_change"] = post_val - pre_val

                # Percent change
                if pre_val != 0:
                    changes[f"{metric}_pct_change"] = ((post_val - pre_val) / pre_val) * 100
                else:
                    changes[f"{metric}_pct_change"] = np.nan

    return changes


def aggregate_results(derived_data_dir: Path,
                      output_path: Path) -> pd.DataFrame:
    """
    Aggregate all subject results into a single CSV.

    Args:
        derived_data_dir: Root directory containing subject folders
        output_path: Path for output CSV

    Returns:
        DataFrame with all results
    """
    all_results = []

    # Find all subject directories
    subject_dirs = sorted([d for d in derived_data_dir.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])

    logger.info(f"Found {len(subject_dirs)} subject directories")

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        row = {"subject_id": subject_id}

        # Load baseline results
        baseline_dir = subject_dir / "ses-Baseline"
        if baseline_dir.exists():
            baseline = load_session_results(baseline_dir)
            baseline_flat = flatten_results(baseline, prefix="pre_")
            row.update(baseline_flat)
            row["has_baseline"] = True
        else:
            row["has_baseline"] = False

        # Load followup results
        followup_dir = subject_dir / "ses-Followup"
        if followup_dir.exists():
            followup = load_session_results(followup_dir)
            followup_flat = flatten_results(followup, prefix="post_")
            row.update(followup_flat)
            row["has_followup"] = True
        else:
            row["has_followup"] = False

        # Compute change metrics if both sessions exist
        if row.get("has_baseline") and row.get("has_followup"):
            changes = compute_change_metrics(
                {k: v for k, v in row.items() if k.startswith("pre_")},
                {k: v for k, v in row.items() if k.startswith("post_")}
            )
            row.update(changes)

        all_results.append(row)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Reorder columns
    column_order = ["subject_id", "has_baseline", "has_followup"]

    # Pre metrics
    pre_cols = [c for c in df.columns if c.startswith("pre_")]
    pre_cols.sort()

    # Post metrics
    post_cols = [c for c in df.columns if c.startswith("post_")]
    post_cols.sort()

    # Change metrics
    change_cols = [c for c in df.columns if c.endswith("_change") or c.endswith("_pct_change")]
    change_cols.sort()

    # Other columns
    other_cols = [c for c in df.columns
                  if c not in column_order + pre_cols + post_cols + change_cols]

    final_order = column_order + pre_cols + post_cols + change_cols + other_cols
    final_order = [c for c in final_order if c in df.columns]

    df = df[final_order]

    # Save CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total subjects: {len(df)}")
    logger.info(f"Subjects with baseline: {df['has_baseline'].sum()}")
    logger.info(f"Subjects with followup: {df['has_followup'].sum()}")
    logger.info(f"Subjects with both: {(df['has_baseline'] & df['has_followup']).sum()}")

    return df


def generate_summary_statistics(df: pd.DataFrame, output_dir: Path):
    """
    Generate summary statistics report.

    Args:
        df: Results DataFrame
        output_dir: Directory for output files
    """
    # Primary endpoints summary
    primary_metrics = [
        "L1L2_vBMD_mean_mgcm3",
        "muscle_SMD_mean_hu",
        "muscle_low_density_percent",
        "IMAT_percent",
        "VAT_volume_cm3",
        "muscle_CSA_VAT_ratio"
    ]

    summary = []

    for metric in primary_metrics:
        pre_col = f"pre_{metric}"
        post_col = f"post_{metric}"
        change_col = f"{metric}_change"

        row = {"metric": metric}

        if pre_col in df.columns:
            pre_values = df[pre_col].dropna()
            row["pre_n"] = len(pre_values)
            row["pre_mean"] = pre_values.mean()
            row["pre_std"] = pre_values.std()
            row["pre_median"] = pre_values.median()

        if post_col in df.columns:
            post_values = df[post_col].dropna()
            row["post_n"] = len(post_values)
            row["post_mean"] = post_values.mean()
            row["post_std"] = post_values.std()
            row["post_median"] = post_values.median()

        if change_col in df.columns:
            change_values = df[change_col].dropna()
            row["change_n"] = len(change_values)
            row["change_mean"] = change_values.mean()
            row["change_std"] = change_values.std()

        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary statistics saved to {summary_path}")

    # Print to console
    print("\n" + "=" * 60)
    print("PRIMARY ENDPOINT SUMMARY")
    print("=" * 60)

    for _, row in summary_df.iterrows():
        metric = row["metric"]
        print(f"\n{metric}:")
        if "pre_mean" in row and pd.notna(row["pre_mean"]):
            print(f"  Baseline: {row['pre_mean']:.2f} +/- {row['pre_std']:.2f} (n={int(row['pre_n'])})")
        if "post_mean" in row and pd.notna(row["post_mean"]):
            print(f"  Followup: {row['post_mean']:.2f} +/- {row['post_std']:.2f} (n={int(row['post_n'])})")
        if "change_mean" in row and pd.notna(row["change_mean"]):
            print(f"  Change:   {row['change_mean']:.2f} +/- {row['change_std']:.2f} (n={int(row['change_n'])})")

    return summary_df


def create_bmd_csv(derived_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Create CSV with BMD-specific metrics only.

    Args:
        derived_dir: Directory containing all DerivedData
        output_path: Path for output CSV

    Returns:
        DataFrame with BMD metrics
    """
    rows = []

    for subject_dir in sorted(derived_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
            continue

        subject_id = subject_dir.name

        for session in ["ses-Baseline", "ses-Followup"]:
            session_dir = subject_dir / session

            if not session_dir.exists():
                continue

            bone_path = session_dir / "bone_results.json"
            cal_path = session_dir / "calibration_bmd.json"

            row = {
                "subject_id": subject_id,
                "session": session.replace("ses-", ""),
            }

            # Load calibration info
            if cal_path.exists():
                with open(cal_path) as f:
                    cal = json.load(f)
                row["calibration_r_squared"] = cal.get("regression", {}).get("r_squared", np.nan)
                row["calibration_slope"] = cal.get("regression", {}).get("slope", np.nan)
                row["calibration_intercept"] = cal.get("regression", {}).get("intercept", np.nan)

            # Load bone results
            if bone_path.exists():
                with open(bone_path) as f:
                    bone = json.load(f)

                row["success"] = bone.get("success", False)
                row["L1L2_vBMD_mean_mgcm3"] = bone.get("L1L2_vBMD_mean_mgcm3")
                row["L1L2_vBMD_weighted_mean_mgcm3"] = bone.get("L1L2_vBMD_weighted_mean_mgcm3")
                row["erosion_mm"] = bone.get("erosion_mm")

                # L1 specific
                if bone.get("L1"):
                    l1 = bone["L1"]
                    row["L1_vBMD_mean_mgcm3"] = l1.get("vBMD_mean_mgcm3")
                    row["L1_vBMD_median_mgcm3"] = l1.get("vBMD_median_mgcm3")
                    row["L1_vBMD_std_mgcm3"] = l1.get("vBMD_std_mgcm3")
                    row["L1_trabecular_volume_cm3"] = l1.get("trabecular_volume_cm3")
                    row["L1_total_body_volume_cm3"] = l1.get("total_body_volume_cm3")
                    row["L1_hu_mean"] = l1.get("hu_mean")

                # L2 specific
                if bone.get("L2"):
                    l2 = bone["L2"]
                    row["L2_vBMD_mean_mgcm3"] = l2.get("vBMD_mean_mgcm3")
                    row["L2_vBMD_median_mgcm3"] = l2.get("vBMD_median_mgcm3")
                    row["L2_vBMD_std_mgcm3"] = l2.get("vBMD_std_mgcm3")
                    row["L2_trabecular_volume_cm3"] = l2.get("trabecular_volume_cm3")
                    row["L2_total_body_volume_cm3"] = l2.get("total_body_volume_cm3")
                    row["L2_hu_mean"] = l2.get("hu_mean")

                row["qc_messages"] = "; ".join(bone.get("qc_messages", []))

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"BMD results saved to {output_path}")
    return df


def create_muscle_csv(derived_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Create CSV with muscle-specific metrics only.

    Args:
        derived_dir: Directory containing all DerivedData
        output_path: Path for output CSV

    Returns:
        DataFrame with muscle metrics
    """
    rows = []

    for subject_dir in sorted(derived_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
            continue

        subject_id = subject_dir.name

        for session in ["ses-Baseline", "ses-Followup"]:
            session_dir = subject_dir / session

            if not session_dir.exists():
                continue

            muscle_path = session_dir / "muscle_results.json"

            row = {
                "subject_id": subject_id,
                "session": session.replace("ses-", ""),
            }

            if muscle_path.exists():
                with open(muscle_path) as f:
                    muscle = json.load(f)

                row["success"] = muscle.get("success", False)
                row["compartment_volume_cm3"] = muscle.get("compartment_volume_cm3")
                row["muscle_tissue_volume_cm3"] = muscle.get("muscle_tissue_volume_cm3")
                row["muscle_normal_volume_cm3"] = muscle.get("muscle_normal_volume_cm3")
                row["muscle_low_density_volume_cm3"] = muscle.get("muscle_low_density_volume_cm3")
                row["imat_volume_cm3"] = muscle.get("imat_volume_cm3")
                row["muscle_low_density_percent"] = muscle.get("muscle_low_density_percent")
                row["imat_percent"] = muscle.get("imat_percent")
                row["muscle_SMD_mean_hu"] = muscle.get("muscle_SMD_mean_hu")
                row["muscle_SMD_median_hu"] = muscle.get("muscle_SMD_median_hu")
                row["muscle_SMD_std_hu"] = muscle.get("muscle_SMD_std_hu")
                row["muscle_SMD_P10_hu"] = muscle.get("muscle_SMD_P10_hu")
                row["muscle_SMD_P90_hu"] = muscle.get("muscle_SMD_P90_hu")
                row["muscle_CSA_mean_cm2"] = muscle.get("muscle_CSA_mean_cm2")
                row["muscle_CSA_max_cm2"] = muscle.get("muscle_CSA_max_cm2")
                row["muscle_LR_symmetry_index"] = muscle.get("muscle_LR_symmetry_index")
                row["drift_correction_hu"] = muscle.get("drift_correction_hu")
                row["n_slices"] = muscle.get("n_slices")
                row["qc_messages"] = "; ".join(muscle.get("qc_messages", []))

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Muscle results saved to {output_path}")
    return df


def create_adipose_csv(derived_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Create CSV with adipose-specific metrics only.

    Args:
        derived_dir: Directory containing all DerivedData
        output_path: Path for output CSV

    Returns:
        DataFrame with adipose metrics
    """
    rows = []

    for subject_dir in sorted(derived_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
            continue

        subject_id = subject_dir.name

        for session in ["ses-Baseline", "ses-Followup"]:
            session_dir = subject_dir / session

            if not session_dir.exists():
                continue

            adipose_path = session_dir / "adipose_results.json"

            row = {
                "subject_id": subject_id,
                "session": session.replace("ses-", ""),
            }

            if adipose_path.exists():
                try:
                    with open(adipose_path) as f:
                        adipose = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted adipose JSON: {adipose_path}")
                    rows.append(row)
                    continue

                row["success"] = adipose.get("success", False)
                row["vat_volume_cm3"] = adipose.get("vat_volume_cm3")
                row["vat_area_L1L2_cm2"] = adipose.get("vat_area_L1L2_cm2")
                row["vat_hu_mean"] = adipose.get("vat_hu_mean")
                row["vat_hu_median"] = adipose.get("vat_hu_median")
                row["vat_outlier_fraction"] = adipose.get("vat_outlier_fraction")
                row["sat_fov_adequate"] = adipose.get("sat_fov_adequate")
                row["sat_volume_cm3"] = adipose.get("sat_volume_cm3")
                row["sat_area_L1L2_cm2"] = adipose.get("sat_area_L1L2_cm2")
                row["sat_hu_mean"] = adipose.get("sat_hu_mean")
                row["sat_hu_median"] = adipose.get("sat_hu_median")
                row["vat_sat_ratio"] = adipose.get("vat_sat_ratio")
                row["muscle_csa_vat_ratio"] = adipose.get("muscle_csa_vat_ratio")
                row["qc_messages"] = "; ".join(adipose.get("qc_messages", []))

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Adipose results saved to {output_path}")
    return df


def create_validation_csv(derived_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Create CSV with tissue validation metrics only.

    Args:
        derived_dir: Directory containing all DerivedData
        output_path: Path for output CSV

    Returns:
        DataFrame with validation metrics
    """
    rows = []

    for subject_dir in sorted(derived_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
            continue

        subject_id = subject_dir.name

        for session in ["ses-Baseline", "ses-Followup"]:
            session_dir = subject_dir / session

            if not session_dir.exists():
                continue

            validation_path = session_dir / "validation_results.json"

            row = {
                "subject_id": subject_id,
                "session": session.replace("ses-", ""),
            }

            if validation_path.exists():
                try:
                    with open(validation_path) as f:
                        validation = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted validation JSON: {validation_path}")
                    rows.append(row)
                    continue

                row["success"] = validation.get("success", False)
                row["imat_dice"] = validation.get("imat_dice")
                row["imat_hu_volume_cm3"] = validation.get("imat_hu_volume_cm3")
                row["imat_ts_in_envelope_cm3"] = validation.get("imat_ts_in_envelope_volume_cm3")
                row["imat_volume_ratio"] = validation.get("imat_volume_ratio")
                row["ts_imat_raw_volume_cm3"] = validation.get("ts_imat_raw_volume_cm3")
                row["muscle_envelope_volume_cm3"] = validation.get("muscle_envelope_volume_cm3")
                row["imat_dice_acceptable"] = validation.get("imat_dice_acceptable")
                row["envelope_coverage_ok"] = validation.get("envelope_coverage_ok")
                row["qc_messages"] = "; ".join(validation.get("qc_messages", []))

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Validation results saved to {output_path}")
    return df


def create_all_separate_csvs(derived_dir: Path, output_dir: Path):
    """
    Create all separate CSV files for BMD, muscle, adipose, and validation.

    Args:
        derived_dir: Directory containing all DerivedData
        output_dir: Directory for output CSV files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    create_bmd_csv(derived_dir, output_dir / "bmd_results.csv")
    create_muscle_csv(derived_dir, output_dir / "muscle_results.csv")
    create_adipose_csv(derived_dir, output_dir / "adipose_results.csv")
    create_validation_csv(derived_dir, output_dir / "validation_results.csv")

    logger.info(f"All separate CSVs created in {output_dir}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python results_aggregation.py <derived_data_dir> <output_dir>")
        sys.exit(1)

    derived_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "results.csv"

    df = aggregate_results(derived_dir, output_path)

    if len(df) > 0:
        generate_summary_statistics(df, output_dir)
