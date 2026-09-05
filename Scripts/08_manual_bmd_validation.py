#!/usr/bin/env python3
"""
Compare automatic pipeline BMD against manual physicist measurements.

Manual data: RawData/bmd_ct/BMD_manual/results.xlsx
  - Sheet 'mean_mgml': per-subject mean trabecular vBMD (mg/cm³)
  - kota1 / kota2: which physical vertebra each refers to is taken from
    Scripts/manual_kota_mapping.csv (per subject, with evidence grade).
    The 2026-09 review showed kota1 is the INFERIOR vertebra (pipeline L2)
    in all 9 subjects with ImageJ screenshots; the earlier assumption
    kota1 = cranial was wrong. The 4 subjects without screenshots use the
    same convention, graded "assumed"; statistics are reported for all 13
    and for the confirmed subset.
  - _b = baseline, _f = follow-up

Pipeline data: Outputs/results.csv
  - pre_L1_vBMD_mean_mgcm3, pre_L2_vBMD_mean_mgcm3, etc.

Outputs (QC/manual_validation/):
  - scatter_L1.png, scatter_L2.png, scatter_L1L2.png
  - bland_altman_L1L2.png
  - comparison_summary.csv
  - validation_report.txt
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MANUAL_XLSX = ROOT.parent / "RawData" / "bmd_ct" / "BMD_manual" / "results.xlsx"
MAPPING_CSV = ROOT / "Scripts" / "manual_kota_mapping.csv"
PIPELINE_CSV = ROOT / "Outputs" / "results.csv"
OUT_DIR = ROOT / "QC" / "manual_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_manual():
    """Load manual BMD from the mean_mgml sheet."""
    df = pd.read_excel(MANUAL_XLSX, sheet_name="mean_mgml", header=0)
    # First column is subject ID (named "Row"), next cols: kota1_b, kota1_f, diff, kota2_b, kota2_f, diff
    df = df.rename(columns={df.columns[0]: "subject_id"})
    # Drop summary rows (Medel, SD, p, blanks)
    df = df[df["subject_id"].astype(str).str.startswith("sub-")].copy()
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    kota = pd.DataFrame({
        "subject_id": df["subject_id"].values,
        "kota1_baseline": pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
        "kota1_followup": pd.to_numeric(df.iloc[:, 2], errors="coerce").values,
        "kota2_baseline": pd.to_numeric(df.iloc[:, 4], errors="coerce").values,
        "kota2_followup": pd.to_numeric(df.iloc[:, 5], errors="coerce").values,
    })
    # Per-subject physical mapping (pipeline L1 = superior body, L2 = inferior body)
    mapping = pd.read_csv(MAPPING_CSV)
    missing = set(kota["subject_id"]) - set(mapping["subject_id"])
    if missing:
        raise ValueError(f"No kota mapping for {sorted(missing)} in {MAPPING_CSV.name}")
    kota = kota.merge(mapping, on="subject_id", how="left")
    k1_inf = kota["kota1_is"].eq("inferior")
    out = pd.DataFrame({"subject_id": kota["subject_id"]})
    for ses in ["baseline", "followup"]:
        out[f"manual_L1_{ses}"] = np.where(k1_inf, kota[f"kota2_{ses}"], kota[f"kota1_{ses}"])
        out[f"manual_L2_{ses}"] = np.where(k1_inf, kota[f"kota1_{ses}"], kota[f"kota2_{ses}"])
    out["kota1_is"] = kota["kota1_is"].values
    out["mapping_evidence"] = kota["evidence"].values
    out["mapping_confirmed"] = kota["evidence"].str.startswith("confirmed").values
    return out


def load_pipeline():
    """Load pipeline BMD results."""
    df = pd.read_csv(PIPELINE_CSV)
    return df[["subject_id",
               "pre_L1_vBMD_mean_mgcm3", "post_L1_vBMD_mean_mgcm3",
               "pre_L2_vBMD_mean_mgcm3", "post_L2_vBMD_mean_mgcm3"]].copy()


def scatter_with_regression(ax, manual, pipeline, label, color="steelblue"):
    """Plot scatter + identity line + regression, return stats."""
    mask = np.isfinite(manual) & np.isfinite(pipeline)
    m, p = manual[mask], pipeline[mask]
    slope, intercept, r, pval, se = stats.linregress(m, p)
    r2 = r ** 2

    # Identity line
    lo = min(m.min(), p.min()) - 5
    hi = max(m.max(), p.max()) + 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="Identity")

    # Regression line
    x_fit = np.linspace(lo, hi, 100)
    ax.plot(x_fit, slope * x_fit + intercept, color="tomato", lw=1.5,
            label=f"y = {slope:.2f}x + {intercept:.1f}")

    # Points
    ax.scatter(m, p, c=color, edgecolors="k", linewidths=0.4, s=50, zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Manual vBMD (mg/cm³)")
    ax.set_ylabel("Pipeline vBMD (mg/cm³)")
    ax.set_title(f"{label}\nR² = {r2:.3f},  n = {len(m)}")
    ax.legend(fontsize=8, loc="upper left")

    bias = np.mean(p - m)
    mad = np.mean(np.abs(p - m))
    return {"label": label, "n": len(m), "R2": r2, "slope": slope,
            "intercept": intercept, "bias": bias, "MAD": mad,
            "RMSE": np.sqrt(np.mean((p - m) ** 2))}


def bland_altman(ax, manual, pipeline, label):
    """Bland-Altman plot (mean vs difference)."""
    mask = np.isfinite(manual) & np.isfinite(pipeline)
    m, p = manual[mask], pipeline[mask]
    mean_vals = (m + p) / 2
    diff_vals = p - m
    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals, ddof=1)
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    ax.scatter(mean_vals, diff_vals, c="steelblue", edgecolors="k", linewidths=0.4, s=50)
    ax.axhline(mean_diff, color="tomato", lw=1.5, label=f"Bias = {mean_diff:+.1f}")
    ax.axhline(loa_upper, color="grey", ls="--", lw=1, label=f"+1.96 SD = {loa_upper:+.1f}")
    ax.axhline(loa_lower, color="grey", ls="--", lw=1, label=f"−1.96 SD = {loa_lower:+.1f}")
    ax.set_xlabel("Mean of Manual & Pipeline (mg/cm³)")
    ax.set_ylabel("Pipeline − Manual (mg/cm³)")
    ax.set_title(f"Bland-Altman: {label}")
    ax.legend(fontsize=8)


def main():
    manual = load_manual()
    pipe = load_pipeline()
    merged = manual.merge(pipe, on="subject_id")
    print(f"Matched {len(merged)} subjects: {list(merged['subject_id'])}")

    # Build arrays for each comparison
    comparisons = {
        "L1 Baseline": ("manual_L1_baseline", "pre_L1_vBMD_mean_mgcm3"),
        "L1 Follow-up": ("manual_L1_followup", "post_L1_vBMD_mean_mgcm3"),
        "L2 Baseline": ("manual_L2_baseline", "pre_L2_vBMD_mean_mgcm3"),
        "L2 Follow-up": ("manual_L2_followup", "post_L2_vBMD_mean_mgcm3"),
    }

    # --- Per-vertebra scatter plots ---
    stats_rows = []

    # L1 scatter (baseline + followup)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (label, (mc, pc)) in zip(axes, [("L1 Baseline", comparisons["L1 Baseline"]),
                                              ("L1 Follow-up", comparisons["L1 Follow-up"])]):
        row = scatter_with_regression(ax, merged[mc].values, merged[pc].values, label)
        stats_rows.append(row)
    fig.suptitle("L1 vBMD: Pipeline vs Manual", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_L1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # L2 scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (label, (mc, pc)) in zip(axes, [("L2 Baseline", comparisons["L2 Baseline"]),
                                              ("L2 Follow-up", comparisons["L2 Follow-up"])]):
        row = scatter_with_regression(ax, merged[mc].values, merged[pc].values, label)
        stats_rows.append(row)
    fig.suptitle("L2 vBMD: Pipeline vs Manual", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_L2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Combined L1+L2 (all 52 data points) ---
    all_manual = np.concatenate([
        merged["manual_L1_baseline"].values,
        merged["manual_L1_followup"].values,
        merged["manual_L2_baseline"].values,
        merged["manual_L2_followup"].values,
    ])
    all_pipe = np.concatenate([
        merged["pre_L1_vBMD_mean_mgcm3"].values,
        merged["post_L1_vBMD_mean_mgcm3"].values,
        merged["pre_L2_vBMD_mean_mgcm3"].values,
        merged["post_L2_vBMD_mean_mgcm3"].values,
    ])
    fig, ax = plt.subplots(figsize=(7, 6.5))
    row = scatter_with_regression(ax, all_manual, all_pipe, "All L1 + L2 (all sessions)", color="steelblue")
    stats_rows.append(row)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_L1L2_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Bland-Altman for combined ---
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bland_altman(ax, all_manual, all_pipe, "All L1 + L2")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bland_altman_L1L2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Change scores comparison ---
    merged["manual_L1_change"] = merged["manual_L1_followup"] - merged["manual_L1_baseline"]
    merged["pipe_L1_change"] = merged["post_L1_vBMD_mean_mgcm3"] - merged["pre_L1_vBMD_mean_mgcm3"]
    merged["manual_L2_change"] = merged["manual_L2_followup"] - merged["manual_L2_baseline"]
    merged["pipe_L2_change"] = merged["post_L2_vBMD_mean_mgcm3"] - merged["pre_L2_vBMD_mean_mgcm3"]
    merged["manual_L1L2_change"] = (merged["manual_L1_change"] + merged["manual_L2_change"]) / 2
    merged["pipe_L1L2_change"] = (merged["pipe_L1_change"] + merged["pipe_L2_change"]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (label, mc, pc) in zip(axes, [
        ("L1 Change", "manual_L1_change", "pipe_L1_change"),
        ("L2 Change", "manual_L2_change", "pipe_L2_change"),
        ("L1-L2 Mean Change", "manual_L1L2_change", "pipe_L1L2_change"),
    ]):
        row = scatter_with_regression(ax, merged[mc].values, merged[pc].values,
                                       f"Δ {label}", color="darkorange")
        stats_rows.append(row)
    fig.suptitle("Change Scores: Pipeline vs Manual", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_change_scores.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Save summary CSV ---
    for row in stats_rows:
        row["subset"] = "all_13"

    # --- Same level-specific comparisons on the screenshot-confirmed subset ---
    conf = merged[merged["mapping_confirmed"]]
    level_pairs = [
        ("L1 Baseline", "manual_L1_baseline", "pre_L1_vBMD_mean_mgcm3"),
        ("L1 Follow-up", "manual_L1_followup", "post_L1_vBMD_mean_mgcm3"),
        ("L2 Baseline", "manual_L2_baseline", "pre_L2_vBMD_mean_mgcm3"),
        ("L2 Follow-up", "manual_L2_followup", "post_L2_vBMD_mean_mgcm3"),
        ("Δ L1 Change", "manual_L1_change", "pipe_L1_change"),
        ("Δ L2 Change", "manual_L2_change", "pipe_L2_change"),
        ("Δ L1-L2 Mean Change", "manual_L1L2_change", "pipe_L1L2_change"),
    ]
    for label, mc, pc in level_pairs:
        m, p = conf[mc].values, conf[pc].values
        slope, intercept, r, _, _ = stats.linregress(m, p)
        stats_rows.append({"label": label, "n": len(m), "R2": r ** 2, "slope": slope,
                           "intercept": intercept, "bias": np.mean(p - m),
                           "MAD": np.mean(np.abs(p - m)), "RMSE": np.sqrt(np.mean((p - m) ** 2)),
                           "subset": "confirmed_mapping_only"})

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT_DIR / "comparison_summary.csv", index=False, float_format="%.4f")

    # --- Print report ---
    n_conf = int(merged["mapping_confirmed"].sum())
    report = []
    report.append("=" * 65)
    report.append("MANUAL vs PIPELINE BMD VALIDATION REPORT")
    report.append("=" * 65)
    report.append(f"Subjects matched: {len(merged)}")
    report.append(f"Manual source: {MANUAL_XLSX.name}")
    report.append(f"Pipeline source: {PIPELINE_CSV.name}")
    report.append(f"Mapping source: {MAPPING_CSV.name}")
    report.append("")
    report.append("Mapping: per subject from manual_kota_mapping.csv. In all subjects kota1 = INFERIOR")
    report.append("body (pipeline L2) and kota2 = SUPERIOR body (pipeline L1).")
    report.append(f"Evidence: screenshot-confirmed in {n_conf}/{len(merged)} subjects; assumed (same convention,")
    report.append("supported by per-slice HU profile matching) in: "
                  + ", ".join(merged.loc[~merged["mapping_confirmed"], "subject_id"]))
    report.append("The L1-L2 mean change comparison does not depend on the mapping.")
    report.append("")
    for subset, title in [("all_13", f"All subjects (n={len(merged)})"),
                          ("confirmed_mapping_only", f"Screenshot-confirmed mapping only (n={n_conf})")]:
        report.append(title)
        report.append(f"{'Comparison':<25} {'n':>3} {'R²':>7} {'Bias':>8} {'MAD':>7} {'RMSE':>7}")
        report.append("-" * 65)
        for _, r in stats_df[stats_df["subset"] == subset].iterrows():
            report.append(f"{r['label']:<25} {int(r['n']):>3} {r['R2']:>7.3f} {r['bias']:>+8.2f} {r['MAD']:>7.2f} {r['RMSE']:>7.2f}")
        report.append("")

    # Per-subject comparison table
    report.append("Per-subject absolute values (mg/cm³); * = mapping assumed, not screenshot-confirmed:")
    report.append(f"{'Subject':<10} {'Man L1b':>8} {'Pip L1b':>8} {'Man L2b':>8} {'Pip L2b':>8} {'Man L1f':>8} {'Pip L1f':>8} {'Man L2f':>8} {'Pip L2f':>8}")
    report.append("-" * 82)
    for _, r in merged.iterrows():
        report.append(
            f"{r['subject_id'] + ('' if r['mapping_confirmed'] else '*'):<10} "
            f"{r['manual_L1_baseline']:>8.1f} {r['pre_L1_vBMD_mean_mgcm3']:>8.1f} "
            f"{r['manual_L2_baseline']:>8.1f} {r['pre_L2_vBMD_mean_mgcm3']:>8.1f} "
            f"{r['manual_L1_followup']:>8.1f} {r['post_L1_vBMD_mean_mgcm3']:>8.1f} "
            f"{r['manual_L2_followup']:>8.1f} {r['post_L2_vBMD_mean_mgcm3']:>8.1f}"
        )
    report.append("")

    # Change scores
    report.append("Per-subject change scores (mg/cm³):")
    report.append(f"{'Subject':<10} {'Man ΔL1':>8} {'Pip ΔL1':>8} {'Man ΔL2':>8} {'Pip ΔL2':>8}")
    report.append("-" * 42)
    for _, r in merged.iterrows():
        report.append(
            f"{r['subject_id']:<10} "
            f"{r['manual_L1_change']:>+8.2f} {r['pipe_L1_change']:>+8.2f} "
            f"{r['manual_L2_change']:>+8.2f} {r['pipe_L2_change']:>+8.2f}"
        )

    report_text = "\n".join(report)
    print(report_text)
    (OUT_DIR / "validation_report.txt").write_text(report_text)

    # Save merged data
    merged.to_csv(OUT_DIR / "merged_manual_pipeline.csv", index=False, float_format="%.4f")

    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
