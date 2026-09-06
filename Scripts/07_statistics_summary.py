#!/usr/bin/env python3
"""
07_statistics_summary.py - Table 1, ancillary numbers, sensitivity table and figures

Everything the manuscript quotes is produced here from ONE input,
Outputs/results.csv (written by 06_results_aggregation.py). Outputs:

    Outputs/table1.csv, table1.md            main pre/post table (Δ, 95% CI, dz, p)
    Outputs/statistics_summary.csv           same, all columns, machine-readable
    Outputs/ancillary_numbers.md, .json      trabecular volumes, calibration/drift/symmetry
                                             ranges, IMAT Dice, slab heights, label mapping
    Outputs/vertebra_label_mapping.csv       which TotalSegmentator label became "L1"/"L2"
    Outputs/sensitivity_table.csv, .md       final vs body-only+click-slice calibration vs
                                             original pipeline (archive_v1_as_reviewed)
    Outputs/Figures/paired_changes.png       per-subject baseline->follow-up lines, 8 outcomes
    Outputs/Figures/change_distribution.png  per-subject Δ with mean and 95% CI

Definitions:
    Δ           mean of (follow-up - baseline)
    95% CI      t-based CI of the mean Δ
    % diff      Δ / baseline mean x 100 (ratio of cohort means, as in the manuscript table)
    % change    mean of the individual percentage changes (reported separately)
    dz          Δ / SD(Δ)  (Cohen's dz)
    p           two-tailed paired t-test; Wilcoxon signed-rank p also given

Usage:
    python Scripts/07_statistics_summary.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Outputs"
FIG = OUT / "Figures"

METRICS = [
    ("L1L2_vBMD_mean_mgcm3", "L1–L2 vBMD (mg/cm³)"),
    ("L1_vBMD_mean_mgcm3", "L1 vBMD (mg/cm³)"),
    ("L2_vBMD_mean_mgcm3", "L2 vBMD (mg/cm³)"),
    ("muscle_SMD_mean_hu", "Muscle SMD (HU)"),
    ("muscle_low_density_percent", "Low-density muscle (%)"),
    ("IMAT_percent", "IMAT (%)"),
    ("muscle_tissue_volume_cm3", "Muscle volume (cm³)"),
    ("muscle_CSA_mean_cm2", "Muscle CSA (cm²)"),
]


def paired(pre: np.ndarray, post: np.ndarray) -> dict:
    m = np.isfinite(pre) & np.isfinite(post)
    pre, post = pre[m], post[m]
    n = int(pre.size)
    d = post - pre
    out = {"n": n, "baseline_mean": pre.mean(), "baseline_sd": pre.std(ddof=1),
           "followup_mean": post.mean(), "followup_sd": post.std(ddof=1),
           "change_mean": d.mean(), "change_sd": d.std(ddof=1)}
    se = out["change_sd"] / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    out["ci_low"], out["ci_high"] = out["change_mean"] - tcrit * se, out["change_mean"] + tcrit * se
    out["pct_diff_of_means"] = 100 * out["change_mean"] / out["baseline_mean"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["pct_change_individual_mean"] = float(np.nanmean(np.where(pre != 0, 100 * d / pre, np.nan)))
    out["dz"] = out["change_mean"] / out["change_sd"] if out["change_sd"] > 0 else np.nan
    t = stats.ttest_rel(post, pre)
    out["t"], out["p"] = float(t.statistic), float(t.pvalue)
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(post, pre).pvalue)
    except ValueError:
        out["wilcoxon_p"] = np.nan
    return out


def table_for(df: pd.DataFrame, metrics=METRICS, prefix_pre="pre_", prefix_post="post_", suffix="") -> pd.DataFrame:
    rows = []
    for key, label in metrics:
        pre_col, post_col = f"{prefix_pre}{key}{suffix}", f"{prefix_post}{key}{suffix}"
        if pre_col not in df.columns or post_col not in df.columns:
            continue
        r = paired(df[pre_col].to_numpy(float), df[post_col].to_numpy(float))
        r.update({"metric_id": key, "outcome": label})
        rows.append(r)
    cols = ["metric_id", "outcome", "n", "baseline_mean", "baseline_sd", "followup_mean", "followup_sd",
            "change_mean", "change_sd", "ci_low", "ci_high", "pct_diff_of_means", "pct_change_individual_mean",
            "dz", "t", "p", "wilcoxon_p"]
    return pd.DataFrame(rows)[cols]


def fmt_p(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def table1_markdown(t: pd.DataFrame, title: str) -> str:
    lines = [f"**{title}**", "",
             "| Outcome | n | Baseline (SD) | Follow-up (SD) | Δ (95% CI) | % diff | d~z~ | p |",
             "|---|:-:|---|---|---|:-:|:-:|:-:|"]
    for _, r in t.iterrows():
        dec = 1 if abs(r.baseline_mean) >= 10 else 2
        lines.append(f"| {r.outcome} | {r.n} | {r.baseline_mean:.{dec}f} ({r.baseline_sd:.{dec}f}) | "
                     f"{r.followup_mean:.{dec}f} ({r.followup_sd:.{dec}f}) | "
                     f"{r.change_mean:+.2f} ({r.ci_low:+.2f}, {r.ci_high:+.2f}) | {r.pct_diff_of_means:+.1f} | "
                     f"{r.dz:+.2f} | {fmt_p(r.p)} |")
    lines += ["", "Values are mean (SD). Δ = mean change (follow-up minus baseline) with 95% confidence interval. "
                  "% diff = Δ relative to the baseline mean. d~z~ = Δ / SD(Δ). p from two-tailed paired t-tests, "
                  "uncorrected for multiple comparisons."]
    return "\n".join(lines)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------------------
def ancillary(df: pd.DataFrame, derived: Path) -> dict:
    """Numbers quoted in the Results text besides Table 1."""
    a = {}

    def both(col):
        return pd.concat([df[f"pre_{col}"], df[f"post_{col}"]]).astype(float)

    for lvl in ["L1", "L2"]:
        v = both(f"{lvl}_trabecular_volume_cm3")
        a[f"{lvl}_trabecular_volume_cm3_mean"], a[f"{lvl}_trabecular_volume_cm3_sd"] = v.mean(), v.std(ddof=1)
        ratio = (df[f"post_{lvl}_trabecular_volume_cm3"] / df[f"pre_{lvl}_trabecular_volume_cm3"])
        a[f"{lvl}_core_volume_followup_to_baseline_ratio_min"], a[f"{lvl}_core_volume_followup_to_baseline_ratio_max"] = ratio.min(), ratio.max()
        r2 = both(f"{lvl}_calibration_r_squared")
        a[f"{lvl}_colocated_calibration_r2_min"], a[f"{lvl}_colocated_calibration_r2_max"] = r2.min(), r2.max()
        n = both(f"{lvl}_calibration_n_slices")
        a[f"{lvl}_calibration_n_slices_min"], a[f"{lvl}_calibration_n_slices_max"] = int(n.min()), int(n.max())
    r2z = both("calibration_r_squared")
    a["click_slice_calibration_r2_min"], a["click_slice_calibration_r2_max"] = r2z.min(), r2z.max()
    d = both("muscle_drift_correction_hu")
    a["drift_offset_hu_min"], a["drift_offset_hu_max"] = d.min(), d.max()
    s = both("muscle_LR_symmetry_index")
    a["symmetry_index_min"], a["symmetry_index_max"] = s.min(), s.max()
    ns = both("muscle_n_slices")
    a["muscle_slab_slices_min"], a["muscle_slab_slices_max"] = int(ns.min()), int(ns.max())
    a["muscle_slab_height_mm_min"], a["muscle_slab_height_mm_max"] = 0.625 * int(ns.min()), 0.625 * int(ns.max())
    a["n_pairs_with_different_slab_height"] = int((df["pre_muscle_n_slices"] != df["post_muscle_n_slices"]).sum())
    dice = both("imat_dice")
    a["imat_dice_mean"], a["imat_dice_min"], a["imat_dice_max"] = dice.mean(), dice.min(), dice.max()
    a["imat_dice_n_ge_0p5"], a["imat_dice_n_sessions"] = int((dice >= 0.5).sum()), int(dice.notna().sum())
    low = pd.concat([df[["subject_id", "pre_imat_dice", "pre_imat_hu_volume_cm3"]].rename(columns=lambda c: c.replace("pre_", "")).assign(session="Baseline"),
                     df[["subject_id", "post_imat_dice", "post_imat_hu_volume_cm3"]].rename(columns=lambda c: c.replace("post_", "")).assign(session="Followup")])
    low = low[low["imat_dice"] < 0.5]
    a["imat_dice_below_0p5_sessions"] = [f"{r.subject_id} {r.session} (Dice {r.imat_dice:.2f}, HU-IMAT {r.imat_hu_volume_cm3:.1f} cm3)" for r in low.itertuples()]
    a["high_myosteatosis_subjects_gt50pct"] = sorted(set(df.loc[df["pre_muscle_low_density_percent"] > 50, "subject_id"]) |
                                                    set(df.loc[df["post_muscle_low_density_percent"] > 50, "subject_id"]))
    a["n_subjects"] = int(len(df))
    a["n_sessions"] = int(df["has_baseline"].sum() + df["has_followup"].sum())

    # label mapping
    rows = []
    for sub in sorted(p.name for p in derived.glob("sub-*")):
        for ses in ["ses-Baseline", "ses-Followup"]:
            p = derived / sub / ses / "vertebral_bodies" / "vertebra_detection.json"
            if p.exists():
                det = json.load(open(p))
                rows.append({"subject": sub, "session": ses.replace("ses-", ""),
                             "superior_body_label": det["l1"]["original_label"],
                             "inferior_body_label": det["l2"]["original_label"]})
    map_df = pd.DataFrame(rows)
    map_df.to_csv(OUT / "vertebra_label_mapping.csv", index=False)
    a["label_mapping_counts"] = {f"{k[0]}/{k[1]}": int(v) for k, v in
                                 map_df.groupby(["superior_body_label", "inferior_body_label"])["subject"].nunique().items()}
    return a


def ancillary_markdown(a: dict) -> str:
    L = ["# Ancillary numbers (generated by 07_statistics_summary.py)", ""]
    L.append(f"- Subjects: {a['n_subjects']}; sessions: {a['n_sessions']}")
    L.append(f"- Trabecular ROI volume, L1: {a['L1_trabecular_volume_cm3_mean']:.1f} ± {a['L1_trabecular_volume_cm3_sd']:.1f} cm³; "
             f"L2: {a['L2_trabecular_volume_cm3_mean']:.1f} ± {a['L2_trabecular_volume_cm3_sd']:.1f} cm³ (26 sessions)")
    L.append(f"- Follow-up/baseline ROI volume ratio: L1 {a['L1_core_volume_followup_to_baseline_ratio_min']:.2f}–{a['L1_core_volume_followup_to_baseline_ratio_max']:.2f}, "
             f"L2 {a['L2_core_volume_followup_to_baseline_ratio_min']:.2f}–{a['L2_core_volume_followup_to_baseline_ratio_max']:.2f}")
    L.append(f"- Co-located calibration R²: L1 {a['L1_colocated_calibration_r2_min']:.4f}–{a['L1_colocated_calibration_r2_max']:.4f}, "
             f"L2 {a['L2_colocated_calibration_r2_min']:.4f}–{a['L2_colocated_calibration_r2_max']:.4f}; "
             f"slices per fit {min(a['L1_calibration_n_slices_min'], a['L2_calibration_n_slices_min'])}–{max(a['L1_calibration_n_slices_max'], a['L2_calibration_n_slices_max'])}")
    L.append(f"- Click-slice calibration R² (reference): {a['click_slice_calibration_r2_min']:.4f}–{a['click_slice_calibration_r2_max']:.4f}")
    L.append(f"- Drift offset applied to muscle HU: {a['drift_offset_hu_min']:+.1f} to {a['drift_offset_hu_max']:+.1f} HU")
    L.append(f"- L/R symmetry index within envelope: {a['symmetry_index_min']:.2f}–{a['symmetry_index_max']:.2f}")
    L.append(f"- Muscle slab: {a['muscle_slab_slices_min']}–{a['muscle_slab_slices_max']} slices "
             f"({a['muscle_slab_height_mm_min']:.1f}–{a['muscle_slab_height_mm_max']:.1f} mm); "
             f"{a['n_pairs_with_different_slab_height']}/13 pairs differ in slab height between visits")
    L.append(f"- IMAT Dice (HU vs TotalSegmentator, within envelope): mean {a['imat_dice_mean']:.2f}, range "
             f"{a['imat_dice_min']:.2f}–{a['imat_dice_max']:.2f}; ≥0.5 in {a['imat_dice_n_ge_0p5']}/{a['imat_dice_n_sessions']} sessions")
    L.append("- Sessions with Dice < 0.5: " + "; ".join(a["imat_dice_below_0p5_sessions"]))
    L.append(f"- Subjects with low-density muscle > 50%: {', '.join(a['high_myosteatosis_subjects_gt50pct']) or 'none'}")
    L.append("- TotalSegmentator label of the superior/inferior target body (subjects): " +
             "; ".join(f"{k}: {v}" for k, v in a["label_mapping_counts"].items()))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------
def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    scen = []
    t = table_for(df); t["scenario"] = "final: body-only masks + co-located calibration + continuous HU classes"; scen.append(t)
    bone_only = [m for m in METRICS if "vBMD" in m[0]]
    t = table_for(df, bone_only, suffix="_z50_calibration"); t["metric_id"] = t["metric_id"]
    t["scenario"] = "body-only masks, click-slice (z0) calibration"; scen.append(t)
    v1 = OUT / "archive_v1_as_reviewed" / "results.csv"
    if v1.exists():
        t = table_for(pd.read_csv(v1)); t["scenario"] = "original pipeline (v1, as reviewed 2026-09-05)"; scen.append(t)
    s = pd.concat(scen, ignore_index=True)
    s["metric_id"] = s["metric_id"].str.replace("_z50_calibration", "", regex=False)
    return s[["scenario"] + [c for c in s.columns if c != "scenario"]]


def sensitivity_markdown(s: pd.DataFrame) -> str:
    L = ["# Sensitivity of Table 1 to the two review-driven method changes", "",
         "| Scenario | Outcome | Δ | 95% CI | dz | p |", "|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.scenario} | {r.outcome} | {r.change_mean:+.2f} | ({r.ci_low:+.2f}, {r.ci_high:+.2f}) | {r.dz:+.2f} | {fmt_p(r.p)} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------
def figures(df: pd.DataFrame, t: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (key, label) in zip(axs.flat, METRICS):
        pre, post = df[f"pre_{key}"].to_numpy(float), df[f"post_{key}"].to_numpy(float)
        for a, b in zip(pre, post):
            ax.plot([0, 1], [a, b], "-o", color="0.6", ms=3, lw=0.8)
        ax.plot([0, 1], [pre.mean(), post.mean()], "-o", color="C3", lw=2.2, ms=6, label="mean")
        r = t[t.metric_id == key].iloc[0]
        ax.set_title(f"{label}\nΔ {r.change_mean:+.2f} ({r.ci_low:+.2f}, {r.ci_high:+.2f}), p = {fmt_p(r.p)}", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "Follow-up"]); ax.set_xlim(-0.3, 1.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Per-subject change (n = 13)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIG / "paired_changes.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (key, label) in zip(axs.flat, METRICS):
        d = (df[f"post_{key}"] - df[f"pre_{key}"]).to_numpy(float)
        r = t[t.metric_id == key].iloc[0]
        ax.axhline(0, color="k", lw=0.8)
        ax.scatter(np.random.default_rng(0).uniform(-0.12, 0.12, d.size), d, color="0.4", s=22, zorder=3)
        ax.errorbar([0.35], [r.change_mean], yerr=[[r.change_mean - r.ci_low], [r.ci_high - r.change_mean]],
                    fmt="o", color="C3", capsize=5, ms=7, zorder=4, label="mean, 95% CI")
        ax.set_xlim(-0.4, 0.7); ax.set_xticks([]); ax.set_title(label, fontsize=10); ax.set_ylabel("follow-up − baseline")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Distribution of individual changes", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIG / "change_distribution.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    results_path = OUT / "results.csv"
    if not results_path.exists():
        raise SystemExit(f"missing {results_path}")
    df = pd.read_csv(results_path)
    df = df[df["has_baseline"] & df["has_followup"]].reset_index(drop=True)
    print(f"{len(df)} complete pairs from {results_path.name}")

    t = table_for(df)
    t.to_csv(OUT / "statistics_summary.csv", index=False, float_format="%.6g")
    t.round(4).to_csv(OUT / "table1.csv", index=False)
    stamp = f"Generated by 07_statistics_summary.py from Outputs/results.csv (git {git_commit()})"
    (OUT / "table1.md").write_text(table1_markdown(t, "Table 1. Pre–post comparison of CT-derived bone and muscle outcomes.") + f"\n\n_{stamp}_\n")
    for _, r in t.iterrows():
        print(f"  {r.outcome:28s} {r.baseline_mean:7.1f} -> {r.followup_mean:7.1f}  Δ {r.change_mean:+6.2f} "
              f"({r.ci_low:+.2f}, {r.ci_high:+.2f})  dz {r.dz:+.2f}  p={r.p:.3f}")

    a = ancillary(df, ROOT / "DerivedData")
    (OUT / "ancillary_numbers.md").write_text(ancillary_markdown(a) + f"\n_{stamp}_\n")
    json.dump(a, open(OUT / "ancillary_numbers.json", "w"), indent=1, default=str)

    s = sensitivity(df)
    s.to_csv(OUT / "sensitivity_table.csv", index=False, float_format="%.6g")
    (OUT / "sensitivity_table.md").write_text(sensitivity_markdown(s) + f"\n_{stamp}_\n")

    figures(df, t)
    print(f"\nWrote table1.csv/.md, statistics_summary.csv, ancillary_numbers.md/.json, sensitivity_table.csv/.md, "
          f"vertebra_label_mapping.csv, Figures/paired_changes.png, Figures/change_distribution.png in {OUT}")


if __name__ == "__main__":
    main()
