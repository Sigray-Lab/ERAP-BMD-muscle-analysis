#!/usr/bin/env python3
"""
11_manuscript_numbers.py - Consolidate every manuscript / supplement number from
its named source file, at full precision, and generate the supplementary draft.

Round-2 review findings R2-F02 / R2-F06: the consolidated JSON had no checked-in
generator and the supplementary draft contained hand-transcribed numbers that
did not match their sources. This script is the only way those files are made.

Inputs (each recorded with its SHA-256 in the output):
    Outputs/results.csv                              Table 1, ancillary, label mapping (via 07)
    Outputs/archive_v1_as_reviewed/results.csv       pre-review scenario for the sensitivity table
    Outputs/sensitivity_per_slice_sessions.csv       calibration-convention sensitivity (10)
    QC/manual_validation/merged_manual_pipeline.csv  physicist comparison, recomputed here (08 logic)
    Outputs/validation_results.csv                   IMAT Dice per session (07_tissue_validation)
    Outputs/rapa_conc_correlations.csv               exposure correlations (09; 4-decimal as written by 09)
    Outputs/sensitivity_per_slice_rapa.csv           exposure correlations per calibration convention (10)
    QC/body_isolation_v2_summary.csv                 edge-contact warnings for the selected bodies

Outputs:
    Outputs/manuscript_numbers.json
    Outputs/supplementary_tables.md
    BMD_muscle_Supplementary_draft.md                (whole file; do not edit by hand)

Usage:
    python Scripts/11_manuscript_numbers.py
"""

import hashlib
import json
import sys
from datetime import date
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
OUT = ROOT / "Outputs"
sys.path.insert(0, str(SCRIPTS))
S7 = import_module("07_statistics_summary")
from utils.provenance import provenance  # noqa: E402

SOURCES = {
    "results": OUT / "results.csv",
    "results_v1": OUT / "archive_v1_as_reviewed" / "results.csv",
    "per_slice_sessions": OUT / "sensitivity_per_slice_sessions.csv",
    "manual_merged": ROOT / "QC" / "manual_validation" / "merged_manual_pipeline.csv",
    "imat_validation": OUT / "validation_results.csv",
    "rapa": OUT / "rapa_conc_correlations.csv",
    "rapa_per_convention": OUT / "sensitivity_per_slice_rapa.csv",
    "body_isolation_summary": ROOT / "QC" / "body_isolation_v2_summary.csv",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def agree(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    f = stats.linregress(x, y)
    d = y - x
    return {"n": int(x.size), "R2": float(f.rvalue ** 2), "slope": float(f.slope), "intercept": float(f.intercept),
            "bias": float(d.mean()), "MAD": float(np.abs(d).mean()), "RMSE": float(np.sqrt((d ** 2).mean())),
            "loa_low": float(d.mean() - 1.96 * d.std(ddof=1)), "loa_high": float(d.mean() + 1.96 * d.std(ddof=1))}


def manual_validation(m: pd.DataFrame) -> list:
    pairs = [("L1 Baseline", "manual_L1_baseline", "pre_L1_vBMD_mean_mgcm3"),
             ("L1 Follow-up", "manual_L1_followup", "post_L1_vBMD_mean_mgcm3"),
             ("L2 Baseline", "manual_L2_baseline", "pre_L2_vBMD_mean_mgcm3"),
             ("L2 Follow-up", "manual_L2_followup", "post_L2_vBMD_mean_mgcm3"),
             ("Δ L1", "manual_L1_change", "pipe_L1_change"),
             ("Δ L2", "manual_L2_change", "pipe_L2_change"),
             ("Δ L1–L2 mean", "manual_L1L2_change", "pipe_L1L2_change")]
    rows = []
    for subset, df in [("all_13", m), ("confirmed_mapping_only", m[m["mapping_confirmed"]])]:
        for label, a, b in pairs:
            r = agree(df[a], df[b]); r.update({"comparison": label, "subset": subset}); rows.append(r)
        allx = np.concatenate([df[c] for c in ["manual_L1_baseline", "manual_L1_followup", "manual_L2_baseline", "manual_L2_followup"]])
        ally = np.concatenate([df[c] for c in ["pre_L1_vBMD_mean_mgcm3", "post_L1_vBMD_mean_mgcm3", "pre_L2_vBMD_mean_mgcm3", "post_L2_vBMD_mean_mgcm3"]])
        r = agree(allx, ally); r.update({"comparison": "All L1 + L2 (all sessions)", "subset": subset}); rows.append(r)
    return rows


def convention_table(ps: pd.DataFrame) -> list:
    w = ps.pivot(index="sub", columns="ses")
    rows = []
    for conv, name in [("click", "click slice (9 slices at z0)"), ("roi", "ROI average (primary)"), ("slice", "per slice (9-slice running mean)")]:
        for key, label in [("L1L2", "L1–L2 vBMD (mg/cm³)"), ("L1", "L1 vBMD (mg/cm³)"), ("L2", "L2 vBMD (mg/cm³)"),
                           ("SMD", "Muscle SMD (HU)"), ("low", "Low-density muscle (%)"), ("IMAT", "IMAT (%)")]:
            r = S7.paired(w[f"{key}_{conv}"]["Baseline"].to_numpy(float), w[f"{key}_{conv}"]["Followup"].to_numpy(float))
            r.update({"convention": name, "outcome": label}); rows.append(r)
    return rows


def md_table(rows, cols, headers, fmt):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    for k, p in SOURCES.items():
        if not p.exists():
            raise SystemExit(f"missing source {k}: {p}")
    res = pd.read_csv(SOURCES["results"]); res = res[res["has_baseline"] & res["has_followup"]].reset_index(drop=True)
    table1 = S7.table_for(res).to_dict(orient="records")
    ancillary = S7.ancillary(res, ROOT / "DerivedData")
    sensitivity = S7.sensitivity(res).to_dict(orient="records")
    conventions = convention_table(pd.read_csv(SOURCES["per_slice_sessions"]))
    manual = manual_validation(pd.read_csv(SOURCES["manual_merged"]))
    imat = pd.read_csv(SOURCES["imat_validation"])
    rapa = pd.read_csv(SOURCES["rapa"]); rapa_conv = pd.read_csv(SOURCES["rapa_per_convention"])
    bodies = pd.read_csv(SOURCES["body_isolation_summary"])
    edge = [f"{r.sub} {r.ses[4:]}: {r.warnings}" for r in bodies.itertuples() if isinstance(r.warnings, str) and "z-edge" in r.warnings]
    label_map = pd.read_csv(OUT / "vertebra_label_mapping.csv")

    doc = {
        "generated": str(date.today()), "generator": "Scripts/11_manuscript_numbers.py", "provenance": provenance(),
        "status": "post-review results; supersede all numbers dated before 2026-09-06",
        "precision_note": ("table1/ancillary/sensitivity/conventions/manual are recomputed here at full float precision; "
                           "rapa correlations are read as written by 09_rapa_correlation.py (4 decimals)."),
        "sources": {k: {"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for k, p in SOURCES.items()},
        "table1": table1, "ancillary": ancillary, "sensitivity_review_changes": sensitivity,
        "sensitivity_calibration_convention": conventions, "manual_validation": manual,
        "rapa_correlations": rapa.to_dict(orient="records"),
        "rapa_correlations_by_convention": rapa_conv.to_dict(orient="records"),
        "imat_validation_per_session": imat[["subject_id", "session", "imat_dice", "imat_hu_volume_cm3", "imat_ts_in_envelope_volume_cm3"]].to_dict(orient="records"),
        "selected_bodies_touching_z_boundary": edge,
        "vertebra_label_mapping": label_map.to_dict(orient="records"),
    }
    json.dump(doc, open(OUT / "manuscript_numbers.json", "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))

    # ------------------------------------------------------------------ supplement
    f1 = lambda v: f"{v:.1f}"; f2 = lambda v: f"{v:+.2f}"; f3 = lambda v: f"{v:.3f}"; fp = S7.fmt_p
    L = []
    L.append("## Supplementary Table S1. Vertebral level selected by the pipeline\n")
    L.append("Source: `Outputs/vertebra_label_mapping.csv`. The two largest whole-vertebra masks in the ~80 mm field of view were selected "
             "as the target vertebrae at both visits; rigid registration supported the same physical pair at both visits "
             "(final body-mask Dice 0.93–0.95). The TotalSegmentator label is unreliable in a narrow field of view and the absolute level was "
             "not verified against a wide-field scan. Three selected body instances touch the acquisition z-boundary (" + "; ".join(e.split(":")[0] for e in edge) +
             "); their trabecular cores are interior.\n")
    cnt = label_map.groupby(["superior_body_label", "inferior_body_label"])["subject"].apply(lambda s: sorted(set(s)))
    L.append("| Superior / inferior body label | Participants |\n|---|---|")
    for (a, b), subs in cnt.items():
        L.append(f"| {a} / {b} | {len(subs)} ({', '.join(subs)}) |")
    L.append("\n## Supplementary Table S2. Sensitivity of Table 1 to the post-review method changes\n")
    L.append("Source: `Outputs/sensitivity_table.csv`. (a) original pipeline (whole-vertebra largest-component masks, click-slice calibration); "
             "(b) body-only masks with the click-slice calibration; (c) final method (body-only masks, per-vertebra ROI-average calibration).\n")
    L.append(md_table(sensitivity, ["scenario", "outcome", "change_mean", "ci_low", "ci_high", "dz", "p"],
                      ["Scenario", "Outcome", "Δ", "CI low", "CI high", "dz", "p"],
                      {"change_mean": f2, "ci_low": f2, "ci_high": f2, "dz": f2, "p": fp}))
    L.append("\n## Supplementary Table S3. Calibration convention (final masks)\n")
    L.append("Source: `Outputs/sensitivity_per_slice_sessions.csv` (Scripts/10). Three ways of using the same manual reference clicks: "
             "click slice only (9 slices), references averaged over the ROI slices (primary), and a separate line per slice from a 9-slice running mean.\n")
    L.append(md_table(conventions, ["convention", "outcome", "change_mean", "ci_low", "ci_high", "change_sd", "dz", "p"],
                      ["Convention", "Outcome", "Δ", "CI low", "CI high", "SD(Δ)", "dz", "p"],
                      {"change_mean": f2, "ci_low": f2, "ci_high": f2, "change_sd": lambda v: f"{v:.2f}", "dz": f2, "p": fp}))
    L.append("\n## Supplementary Table S4. Pipeline versus separately calibrated manual vBMD\n")
    L.append("Source: `QC/manual_validation/merged_manual_pipeline.csv` (Scripts/08), statistics recomputed here. Manual measurements by a medical physicist "
             "on the 2.5 mm reconstruction (anterior elliptical ROIs, 9–13 central slices, separate manual phantom calibration). The manual \"kota1\" is the "
             "inferior body (screenshot-confirmed in 9/13, assumed in 4). Bias = pipeline − manual.\n")
    L.append(md_table(manual, ["subset", "comparison", "n", "R2", "bias", "RMSE", "loa_low", "loa_high"],
                      ["Subset", "Comparison", "n", "R²", "Bias", "RMSE (mg/cm³)", "LoA low", "LoA high"],
                      {"R2": f3, "bias": f2, "RMSE": lambda v: f"{v:.2f}", "loa_low": f2, "loa_high": f2}))
    L.append("\nFigures: `QC/manual_validation/scatter_L1.png`, `scatter_L2.png`, `scatter_L1L2_combined.png`, `bland_altman_L1L2.png`, `scatter_change_scores.png`. "
             "No manual value enters the pipeline; the method revisions and this comparison used the same cohort, so it is not a held-out validation "
             "and does not establish absolute phantom accuracy.\n")
    L.append("## Supplementary Table S5. IMAT: HU-based versus TotalSegmentator tissue_4_types (within the muscle envelope)\n")
    d = imat["imat_dice"]
    low = imat[imat["imat_dice"] < 0.5]
    L.append(f"Source: `Outputs/validation_results.csv`. Mean Dice {d.mean():.2f} (range {d.min():.2f}–{d.max():.2f}); Dice ≥ 0.5 in {(d >= 0.5).sum()}/{len(d)} sessions. "
             "Sessions below 0.5: " + "; ".join(f"{r.subject_id} {r.session[4:]} ({r.imat_dice:.2f}, HU-IMAT {r.imat_hu_volume_cm3:.1f} cm³)" for r in low.itertuples()) +
             ". Where the HU-based IMAT volume is 3–5 cm³ the disagreement is not explained solely by small volume.\n")
    L.append("## Supplementary Table S6. Rapamycin concentration (48 h) versus change in CT outcomes\n")
    prim = rapa[rapa["analysis"].str.startswith("primary")]
    L.append(f"Source: `Outputs/rapa_conc_correlations.csv` (Scripts/09); figure `Outputs/Figures/rapa_conc_vs_change.png`. Primary n = {int(prim['n'].iloc[0])} "
             "(sub-104 and sub-107 excluded: concentration values flagged as erroneous in the trial data curation); n = 13 sensitivity rows in the CSV.\n")
    L.append(md_table(prim.to_dict(orient="records"), ["outcome", "pearson_r", "p_value", "spearman_rho", "spearman_p"],
                      ["Δ outcome", "Pearson r", "p", "Spearman ρ", "p"],
                      {"pearson_r": f2, "p_value": fp, "spearman_rho": f2, "spearman_p": fp}))
    smd = rapa_conv[rapa_conv["outcome"].str.contains("SMD")].set_index("convention")
    L.append(f"\nThe SMD correlation depends on the base-material adjustment convention: ROI average r = {smd.loc['roi','pearson_r']:+.2f} (p = {fp(smd.loc['roi','p'])}), "
             f"tracked click window r = {smd.loc['click','pearson_r']:+.2f} (p = {fp(smd.loc['click','p'])}), per slice r = {smd.loc['slice','pearson_r']:+.2f} "
             f"(p = {fp(smd.loc['slice','p'])}) (`Outputs/sensitivity_per_slice_rapa.csv`). Muscle volume depends on slab height; mean CSA is less directly "
             "dependent on slab height but still depends on the slices included.\n")
    L.append("## Supplementary Figures (QC of the post-review method)\n")
    L.append("| Figure | File | Shows |\n|---|---|---|")
    for fig, f, what in [
        ("S1", "`QC/phantom_tracking_extremes_1.png`, `_2.png`", "Tracked 4 mm sampling circle on every reference at the first and last slice of all 26 scans"),
        ("S2", "`QC/phantom_tracking_rod_boundary_check.png`", "Circle position versus the rod boundary detected from the image (image-derived margin ≥ 2.1 mm with the threshold edge definition)"),
        ("S3", "`QC/phantom_tracking_slab_extremes_1.png`, `_2.png`", "Same at the bottom and top slice of every muscle slab (tight window)"),
        ("S4", "`QC/phantom_base_slab_stability.png`", "Base-material HU along z for all scans, click window versus muscle slab"),
        ("S5", "`QC/sub-103/ses-Baseline/phantom_tracking.png` (example)", "Per-scan tracking sheet: references × slices, drift, HU, noise and edge profiles"),
        ("S6", "`QC/body_isolation_v2_sagittal_1.png`, `_2.png`", "Original versus body-only vertebral masks and cores, all scans"),
        ("S7", "`QC/sub-113/ses-Baseline/body_isolation_v2.png`, `QC/sub-114/ses-Baseline/body_isolation_v2.png`", "Examples: fragmented label repaired; Z-split subject"),
        ("S8", "`Outputs/Figures/paired_changes.png`, `change_distribution.png`", "Per-subject changes for the eight outcomes")]:
        L.append(f"| {fig} | {f} | {what} |")
    a = ancillary
    L.append("\n## Supplementary Methods notes\n")
    L.append(f"- Erosion 5 mm (distance transform), HU filter −50 to 400 applied on top of the saved eroded mask, endplate exclusion 10 % of the body height. "
             f"Trabecular ROI {a['L1_trabecular_volume_cm3_mean']:.1f} ± {a['L1_trabecular_volume_cm3_sd']:.1f} (L1) and {a['L2_trabecular_volume_cm3_mean']:.1f} ± {a['L2_trabecular_volume_cm3_sd']:.1f} (L2) cm³; "
             f"follow-up/baseline ROI volume ratio {min(a['L1_core_volume_followup_to_baseline_ratio_min'], a['L2_core_volume_followup_to_baseline_ratio_min']):.2f}–{max(a['L1_core_volume_followup_to_baseline_ratio_max'], a['L2_core_volume_followup_to_baseline_ratio_max']):.2f}.")
    L.append("- Four visible rod inserts (nominal −100, 50, 100, 200 mg/cm³) and a base-material site (0) sampled with 4 mm-radius circles; tray in-plane displacement along the scan ≤ 2.9 mm; "
             "image-derived rod radius 8.3–9.4 mm (threshold edge) or 9.6–10.3 mm (gradient edge, reviewer), not a certified dimension.")
    L.append(f"- Per-vertebra calibration from references averaged over {min(a['L1_calibration_n_slices_min'], a['L2_calibration_n_slices_min'])}–{max(a['L1_calibration_n_slices_max'], a['L2_calibration_n_slices_max'])} ROI slices; "
             f"R² {min(a['L1_colocated_calibration_r2_min'], a['L2_colocated_calibration_r2_min']):.4f}–{max(a['L1_colocated_calibration_r2_max'], a['L2_colocated_calibration_r2_max']):.4f}.")
    L.append(f"- Muscle envelope: 7-voxel ball closing (≈4.8 mm in-plane, 4.4 mm axial) + 2D hole fill over the endplate-excluded-body span; slab {a['muscle_slab_height_mm_min']:.1f}–{a['muscle_slab_height_mm_max']:.1f} mm; "
             f"slab height differs between visits in {a['n_pairs_with_different_slab_height']}/13 (volume = CSA × height).")
    L.append(f"- Base-material HU adjustment {a['drift_offset_hu_min']:+.1f} to {a['drift_offset_hu_max']:+.1f} HU; individual ΔSMD depends on the adjustment convention (Table S3, S6 note).")
    L.append(f"- Symmetry index within the envelope {a['symmetry_index_min']:.2f}–{a['symmetry_index_max']:.2f}. Low-density muscle > 50 % in {', '.join(a['high_myosteatosis_subjects_gt50pct'])}.")
    tables_md = "\n".join(L) + "\n"
    (OUT / "supplementary_tables.md").write_text(tables_md)
    head = ("# Supplementary material — CT bone and muscle outcomes (generated draft)\n\n"
            f"Generated {date.today()} by `Scripts/11_manuscript_numbers.py` from the source files listed in `Outputs/manuscript_numbers.json` "
            "(git tag v2-review-response and later). Do not edit numbers by hand; re-run the generator.\n\n")
    (ROOT / "BMD_muscle_Supplementary_draft.md").write_text(head + tables_md)
    print("wrote Outputs/manuscript_numbers.json, Outputs/supplementary_tables.md, BMD_muscle_Supplementary_draft.md")


if __name__ == "__main__":
    main()
