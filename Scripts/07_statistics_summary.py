#!/usr/bin/env python3
"""
Generate paired t-test statistics summary for ERAP CT analysis.

Outputs a CSV with: Metric, N, Baseline_Mean, Baseline_SD, Followup_Mean, Followup_SD,
                    Change_Mean, Change_SD, Pct_Change_Mean, P_Value

Usage:
    python Scripts/07_statistics_summary.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


# Metrics to analyze (in order of relevance)
METRICS = [
    # Primary bone endpoint
    ("L1L2_vBMD_mean_mgcm3", "L1L2 vBMD (mg/cm³)", "pre_L1L2_vBMD_mean_mgcm3", "post_L1L2_vBMD_mean_mgcm3"),

    # Muscle quality
    ("muscle_SMD_mean_hu", "Muscle SMD (HU)", "pre_muscle_SMD_mean_hu", "post_muscle_SMD_mean_hu"),
    ("muscle_low_density_percent", "Low-density muscle (%)", "pre_muscle_low_density_percent", "post_muscle_low_density_percent"),
    ("IMAT_percent", "IMAT (%)", "pre_IMAT_percent", "post_IMAT_percent"),

    # Adipose (VAT - may be 0 if tissue_types not run)
    ("VAT_volume_cm3", "VAT volume (cm³)", "pre_VAT_volume_cm3", "post_VAT_volume_cm3"),

    # Muscle volume/size
    ("muscle_tissue_volume_cm3", "Muscle volume (cm³)", "pre_muscle_tissue_volume_cm3", "post_muscle_tissue_volume_cm3"),
    ("muscle_CSA_mean_cm2", "Muscle CSA (cm²)", "pre_muscle_CSA_mean_cm2", "post_muscle_CSA_mean_cm2"),

    # Individual vertebrae
    ("L1_vBMD_mean_mgcm3", "L1 vBMD (mg/cm³)", "pre_L1_vBMD_mean_mgcm3", "post_L1_vBMD_mean_mgcm3"),
    ("L2_vBMD_mean_mgcm3", "L2 vBMD (mg/cm³)", "pre_L2_vBMD_mean_mgcm3", "post_L2_vBMD_mean_mgcm3"),

    # Trabecular volumes
    ("L1_trabecular_volume_cm3", "L1 trabecular vol (cm³)", "pre_L1_trabecular_volume_cm3", "post_L1_trabecular_volume_cm3"),
    ("L2_trabecular_volume_cm3", "L2 trabecular vol (cm³)", "pre_L2_trabecular_volume_cm3", "post_L2_trabecular_volume_cm3"),
]


def compute_paired_stats(df: pd.DataFrame, pre_col: str, post_col: str) -> dict:
    """
    Compute paired t-test statistics for a metric.

    Returns dict with N, means, SDs, change, pct_change, p-value.
    """
    # Get paired data (drop rows with NaN in either column)
    valid = df[[pre_col, post_col]].dropna()
    pre = valid[pre_col]
    post = valid[post_col]

    n = len(pre)

    if n < 2:
        return {
            'N': n,
            'Baseline_Mean': np.nan,
            'Baseline_SD': np.nan,
            'Followup_Mean': np.nan,
            'Followup_SD': np.nan,
            'Change_Mean': np.nan,
            'Change_SD': np.nan,
            'Pct_Change_Mean': np.nan,
            'P_Value': np.nan
        }

    # Basic stats
    baseline_mean = pre.mean()
    baseline_sd = pre.std()
    followup_mean = post.mean()
    followup_sd = post.std()

    # Change
    change = post - pre
    change_mean = change.mean()
    change_sd = change.std()

    # Percent change (relative to baseline)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_change = np.where(pre != 0, (change / pre) * 100, np.nan)
    pct_change_mean = np.nanmean(pct_change)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(pre, post)

    return {
        'N': n,
        'Baseline_Mean': baseline_mean,
        'Baseline_SD': baseline_sd,
        'Followup_Mean': followup_mean,
        'Followup_SD': followup_sd,
        'Change_Mean': change_mean,
        'Change_SD': change_sd,
        'Pct_Change_Mean': pct_change_mean,
        'P_Value': p_value
    }


def main():
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load results
    results_path = project_root / "Outputs" / "results.csv"
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} subjects from {results_path}")

    # Compute statistics for each metric
    stats_data = []
    for metric_id, metric_name, pre_col, post_col in METRICS:
        if pre_col not in df.columns or post_col not in df.columns:
            print(f"  Skipping {metric_name}: columns not found")
            continue

        result = compute_paired_stats(df, pre_col, post_col)
        result['Metric_ID'] = metric_id
        result['Metric_Name'] = metric_name
        stats_data.append(result)

        # Print summary
        sig = "*" if result['P_Value'] < 0.05 else ""
        print(f"  {metric_name}: {result['Baseline_Mean']:.2f} → {result['Followup_Mean']:.2f}, "
              f"Δ={result['Change_Mean']:+.2f} ({result['Pct_Change_Mean']:+.1f}%), p={result['P_Value']:.4f}{sig}")

    # Create DataFrame
    stats_df = pd.DataFrame(stats_data)

    # Reorder columns
    cols = ['Metric_ID', 'Metric_Name', 'N', 'Baseline_Mean', 'Baseline_SD',
            'Followup_Mean', 'Followup_SD', 'Change_Mean', 'Change_SD',
            'Pct_Change_Mean', 'P_Value']
    stats_df = stats_df[cols]

    # Round numeric columns
    for col in ['Baseline_Mean', 'Baseline_SD', 'Followup_Mean', 'Followup_SD',
                'Change_Mean', 'Change_SD', 'Pct_Change_Mean']:
        stats_df[col] = stats_df[col].round(2)
    stats_df['P_Value'] = stats_df['P_Value'].round(4)

    # Save to CSV
    output_path = project_root / "Outputs" / "statistics_summary.csv"
    stats_df.to_csv(output_path, index=False)
    print(f"\nStatistics saved to: {output_path}")

    # Also save a formatted markdown table
    md_path = project_root / "Outputs" / "statistics_summary.md"
    with open(md_path, 'w') as f:
        f.write("# ERAP CT Analysis - Paired T-Test Results\n\n")
        f.write(f"N = {stats_df['N'].iloc[0]} subjects (paired baseline/followup)\n\n")
        f.write("| Metric | Baseline (mean±SD) | Followup (mean±SD) | Change | % Change | p-value |\n")
        f.write("|--------|-------------------|-------------------|--------|----------|--------|\n")

        for _, row in stats_df.iterrows():
            baseline = f"{row['Baseline_Mean']:.1f} ± {row['Baseline_SD']:.1f}"
            followup = f"{row['Followup_Mean']:.1f} ± {row['Followup_SD']:.1f}"
            change = f"{row['Change_Mean']:+.2f}"
            pct = f"{row['Pct_Change_Mean']:+.1f}%"
            p = f"{row['P_Value']:.4f}"
            if row['P_Value'] < 0.05:
                p += " *"
            f.write(f"| {row['Metric_Name']} | {baseline} | {followup} | {change} | {pct} | {p} |\n")

        f.write("\n\\* p < 0.05\n")

    print(f"Markdown table saved to: {md_path}")

    return 0


if __name__ == "__main__":
    exit(main())
