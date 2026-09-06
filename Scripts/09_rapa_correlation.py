#!/usr/bin/env python3
"""
09_rapa_correlation.py - Rapamycin blood concentration vs change in CT outcomes

Reproducible replacement for the 2026-03-02 figure that had no script in this
repository (adversarial review finding F15).

Exposure: rapa_conc_48h from the trial outcomes file (one value per subject).
Exclusions: sub-104 and sub-107 have concentration values flagged as erroneous
in the trial outcome curation and are excluded in every ERAP sibling analysis
(ERAP_cardiovascular_heart/Scripts/10_rapa_correlation.py,
ONH_Analysis/Scripts/statistical_analysis.py). The same exclusion is applied
here; a 13-subject row set is written as a sensitivity, not as the result.

Outputs:
    Outputs/rapa_conc_correlations.csv   Pearson r, p and Spearman rho, p per outcome (n=11 and n=13)
    Outputs/Figures/rapa_conc_vs_change.png
    Outputs/rapa_conc_manifest.json      source file, sha256, exclusions

Usage:
    python Scripts/09_rapa_correlation.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
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
OUTCOMES_CSV = Path("/Users/pontusps/Documents/ERAP_backupdata/BIDS_20260205/raw/All_outcomes_20250630.csv")
EXCLUDE = {"sub-104": "erroneous concentration value (trial data curation)",
           "sub-107": "erroneous concentration value (trial data curation)"}

OUTCOMES = [
    ("L1L2_vBMD_mean_mgcm3_change", "ΔL1–L2 vBMD (mg/cm³)"),
    ("L1_vBMD_mean_mgcm3_change", "ΔL1 vBMD (mg/cm³)"),
    ("L2_vBMD_mean_mgcm3_change", "ΔL2 vBMD (mg/cm³)"),
    ("muscle_SMD_mean_hu_change", "ΔMuscle SMD (HU)"),
    ("muscle_low_density_percent_change", "ΔLow-density muscle (%)"),
    ("IMAT_percent_change", "ΔIMAT (%)"),
    ("muscle_tissue_volume_cm3_change", "ΔMuscle volume (cm³)"),
    ("muscle_CSA_mean_cm2_change", "ΔMuscle CSA (cm²)"),
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_exposure() -> pd.DataFrame:
    df = pd.read_csv(OUTCOMES_CSV)
    df = df.drop_duplicates(subset="Subject")[["Subject", "rapa_conc_48h"]].copy()
    df["subject_id"] = "sub-" + df["Subject"].astype(str)
    return df[["subject_id", "rapa_conc_48h"]]


def correlations(m: pd.DataFrame, label: str) -> list:
    rows = []
    for col, name in OUTCOMES:
        sub = m.dropna(subset=[col, "rapa_conc_48h"])
        r, p = stats.pearsonr(sub["rapa_conc_48h"], sub[col])
        rho, ps = stats.spearmanr(sub["rapa_conc_48h"], sub[col])
        rows.append({"analysis": label, "outcome": name, "column": col, "n": len(sub),
                     "pearson_r": round(r, 4), "p_value": round(p, 4),
                     "spearman_rho": round(rho, 4), "spearman_p": round(ps, 4)})
    return rows


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(OUT / "results.csv")
    res = res[res["has_baseline"] & res["has_followup"]]
    expo = load_exposure()
    merged = res.merge(expo, on="subject_id", how="inner")
    primary = merged[~merged["subject_id"].isin(EXCLUDE)]
    rows = correlations(primary, f"primary (n={len(primary)}, excl. {', '.join(sorted(EXCLUDE))})")
    rows += correlations(merged, f"sensitivity: all subjects with exposure (n={len(merged)})")
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT / "rapa_conc_correlations.csv", index=False)
    print(corr.to_string(index=False))

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (col, name) in zip(axs.flat, OUTCOMES):
        ax.scatter(primary["rapa_conc_48h"], primary[col], color="C0", s=35, zorder=3)
        ex = merged[merged["subject_id"].isin(EXCLUDE)]
        ax.scatter(ex["rapa_conc_48h"], ex[col], facecolors="none", edgecolors="0.5", s=35, label="excluded")
        x = primary["rapa_conc_48h"].to_numpy(float); y = primary[col].to_numpy(float)
        b, a0 = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50); ax.plot(xx, b * xx + a0, color="C3", lw=1.2)
        r = corr[(corr.column == col) & corr.analysis.str.startswith("primary")].iloc[0]
        ax.set_title(f"{name}\nr = {r.pearson_r:.2f}, p = {r.p_value:.3f} (n = {r.n})", fontsize=9)
        ax.set_xlabel("rapamycin conc. 48 h"); ax.axhline(0, color="k", lw=0.5)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Change in CT outcomes vs rapamycin concentration; open circles = excluded ({', '.join(sorted(EXCLUDE))})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "rapa_conc_vs_change.png", dpi=150); plt.close(fig)

    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    json.dump({"timestamp": datetime.now().isoformat(), "git_commit": commit,
               "exposure_file": str(OUTCOMES_CSV), "exposure_sha256": sha256(OUTCOMES_CSV),
               "exposure_column": "rapa_conc_48h", "results_file": "Outputs/results.csv",
               "excluded": EXCLUDE, "n_primary": int(len(primary)), "n_all": int(len(merged))},
              open(OUT / "rapa_conc_manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
