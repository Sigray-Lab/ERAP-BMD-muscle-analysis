#!/usr/bin/env python3
"""
10_per_slice_calibration_sensitivity.py - Per-slice phantom calibration (sensitivity)

Compares three calibration conventions on the FINAL masks (body-only bodies,
current envelope), for every session:

    click     one line / one offset from the 9 slices around the click slice (pre-review)
    roi       one line per vertebra / one offset per muscle slab, rods averaged over the
              ROI slices (current primary)
    slice     a line / offset per slice from the 9-slice running mean of the rods at that
              slice; each voxel is converted with the line of its own slice, then the ROI
              mean is taken

Bone: L1, L2, L1L2 vBMD. Muscle: SMD, low-density %, IMAT % (classes recomputed
with the per-slice offset). Outputs:
    Outputs/sensitivity_per_slice_sessions.csv   per session, all three conventions
    Outputs/sensitivity_per_slice_table1.csv/.md  paired stats under each convention
    Outputs/sensitivity_per_slice_manual.csv      agreement with the physicist per convention
    Outputs/sensitivity_per_slice_rapa.csv        rapamycin correlation per convention

ADDITIVE: no pipeline result is modified.
"""

import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
muscle = import_module("04_muscle_analysis")
statsum = import_module("07_statistics_summary")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Outputs"
RAW = ROOT.parent / "RawData" / "bmd_ct"
REG = ["base", "bone_50", "bone_100", "bone_200"]
DENS = np.array([0.0, 50.0, 100.0, 200.0])
HU_MIN, HU_MAX = -50, 400


def rods(zp):
    return {k: np.array([np.nan if v is None else v for v in zp["per_slice"][f"{k}_mean_hu"]], float) for k in REG}


def lines_per_slice(r):
    """slope[z], intercept[z] from the 9-slice running mean of each rod."""
    sm = {k: uniform_filter1d(np.nan_to_num(v, nan=np.nanmean(v)), 9, mode="nearest") for k, v in r.items()}
    nz = len(sm["base"]); slope = np.zeros(nz); inter = np.zeros(nz)
    for z in range(nz):
        hu = np.array([sm[k][z] for k in REG]); f = stats.linregress(hu, DENS)
        slope[z], inter[z] = f.slope, f.intercept
    return slope, inter, sm["base"]


def line_for(r, zs):
    hu = np.array([np.nanmean(r[k][zs]) for k in REG]); f = stats.linregress(hu, DENS)
    return f.slope, f.intercept


def session(sub, ses):
    d = ROOT / "DerivedData" / sub / ses
    zp = json.load(open(d / "phantom_zprofile.json")); r = rods(zp); z0 = zp["click_slice_z0"]
    ct = np.asarray(nib.load(next((RAW / sub / ses / "ct").glob("*stnd1.25mm_ct.nii.gz"))).dataobj, dtype=np.float32)
    s_sl, i_sl, base_sm = lines_per_slice(r)
    click_zs = np.arange(z0 - 4, z0 + 5); s_ck, i_ck = line_for(r, click_zs)
    row = {"sub": sub, "ses": ses[4:]}
    # ---- bone
    for lvl in ["L1", "L2"]:
        m = np.asarray(nib.load(d / "vertebral_bodies" / "trabecular_masks" / f"{lvl}_trabecular.nii.gz").dataobj) > 0
        hu = ct[m]; zz = np.where(m)[2]; ok = (hu >= HU_MIN) & (hu <= HU_MAX); hu, zz = hu[ok], zz[ok]
        s_roi, i_roi = line_for(r, np.unique(zz))
        row[f"{lvl}_click"] = float(np.mean(s_ck * hu + i_ck))
        row[f"{lvl}_roi"] = float(np.mean(s_roi * hu + i_roi))
        row[f"{lvl}_slice"] = float(np.mean(s_sl[zz] * hu + i_sl[zz]))
    for c in ["click", "roi", "slice"]:
        row[f"L1L2_{c}"] = (row[f"L1_{c}"] + row[f"L2_{c}"]) / 2
    # ---- muscle
    env = np.asarray(nib.load(d / "muscle_compartment.nii.gz").dataobj) > 0
    zs = np.where(env.any(axis=(0, 1)))[0]
    offsets = {"click": -float(np.mean(r["base"][click_zs])), "roi": -float(np.nanmean(r["base"][zs]))}
    for c, off in offsets.items():
        tm = muscle.classify_voxels(ct, env, off)
        row.update(_muscle_metrics(ct + off, env, tm, c))
    off_z = -base_sm                                   # per-slice offset
    corrected = ct + off_z[None, None, :]
    tm = muscle.classify_voxels(corrected, env, 0.0)
    row.update(_muscle_metrics(corrected, env, tm, "slice"))
    return row


def _muscle_metrics(corr, env, tm, tag):
    n_env = env.sum(); n_all = tm["muscle_all"].sum()
    return {f"SMD_{tag}": float(corr[tm["muscle_all"]].mean()),
            f"low_{tag}": 100 * tm["muscle_low"].sum() / n_all,
            f"IMAT_{tag}": 100 * tm["imat"].sum() / n_env}


def main():
    rows = [session(p.parent.name, p.name) for p in sorted(ROOT.glob("DerivedData/sub-*/ses-*")) if (p / "phantom_zprofile.json").exists()]
    df = pd.DataFrame(rows); df.to_csv(OUT / "sensitivity_per_slice_sessions.csv", index=False, float_format="%.4f")
    w = df.pivot(index="sub", columns="ses")
    metrics = [("L1L2", "L1–L2 vBMD (mg/cm³)"), ("L1", "L1 vBMD (mg/cm³)"), ("L2", "L2 vBMD (mg/cm³)"),
               ("SMD", "Muscle SMD (HU)"), ("low", "Low-density muscle (%)"), ("IMAT", "IMAT (%)")]
    t1 = []
    for c, name in [("click", "click slice (9 slices at z0)"), ("roi", "ROI average (current primary)"), ("slice", "per slice (9-slice running mean)")]:
        for key, label in metrics:
            r = statsum.paired(w[f"{key}_{c}"]["Baseline"].to_numpy(float), w[f"{key}_{c}"]["Followup"].to_numpy(float))
            r.update({"convention": name, "outcome": label}); t1.append(r)
    t1 = pd.DataFrame(t1); t1.to_csv(OUT / "sensitivity_per_slice_table1.csv", index=False, float_format="%.5g")
    md = ["# Calibration convention sensitivity (final masks)", "", "| Convention | Outcome | Baseline | Follow-up | Δ (95% CI) | SD(Δ) | dz | p |", "|---|---|---|---|---|---|---|---|"]
    for _, r in t1.iterrows():
        md.append(f"| {r.convention} | {r.outcome} | {r.baseline_mean:.1f} | {r.followup_mean:.1f} | {r.change_mean:+.2f} ({r.ci_low:+.2f}, {r.ci_high:+.2f}) | {r.change_sd:.2f} | {r.dz:+.2f} | {statsum.fmt_p(r.p)} |")
    # manual agreement
    man = pd.read_csv(ROOT / "QC" / "manual_validation" / "merged_manual_pipeline.csv").set_index("subject_id")
    mrows = []
    for c in ["click", "roi", "slice"]:
        for lvl in ["L1", "L2"]:
            for ses, mses in [("Baseline", "baseline"), ("Followup", "followup")]:
                x = man.loc[w.index, f"manual_{lvl}_{mses}"].to_numpy(float); y = w[f"{lvl}_{c}"][ses].to_numpy(float)
                mrows.append(dict(convention=c, comparison=f"{lvl} {ses}", n=len(x), R2=stats.linregress(x, y).rvalue ** 2, bias=np.mean(y - x), RMSE=np.sqrt(np.mean((y - x) ** 2))))
        mx = ((man.loc[w.index, "manual_L1_followup"] - man.loc[w.index, "manual_L1_baseline"]) + (man.loc[w.index, "manual_L2_followup"] - man.loc[w.index, "manual_L2_baseline"])) / 2
        my = w[f"L1L2_{c}"]["Followup"] - w[f"L1L2_{c}"]["Baseline"]
        mrows.append(dict(convention=c, comparison="L1-L2 mean change", n=len(mx), R2=stats.linregress(mx, my).rvalue ** 2, bias=np.mean(my - mx), RMSE=np.sqrt(np.mean((my - mx) ** 2))))
    mdf = pd.DataFrame(mrows); mdf.to_csv(OUT / "sensitivity_per_slice_manual.csv", index=False, float_format="%.4f")
    md += ["", "## Agreement with the physicist's manual vBMD", "", "| Convention | Comparison | R² | bias | RMSE |", "|---|---|---|---|---|"]
    for _, r in mdf[mdf.comparison.isin(["L1-L2 mean change"]) | mdf.comparison.str.contains("Baseline")].iterrows():
        md.append(f"| {r.convention} | {r.comparison} | {r.R2:.3f} | {r.bias:+.2f} | {r.RMSE:.2f} |")
    # rapa
    expo = pd.read_csv("/Users/pontusps/Documents/ERAP_backupdata/BIDS_20260205/raw/All_outcomes_20250630.csv").drop_duplicates("Subject")
    expo["sub"] = "sub-" + expo.Subject.astype(str); expo = expo.set_index("sub")["rapa_conc_48h"]
    keep = [s for s in w.index if s not in ("sub-104", "sub-107")]
    rrows = []
    for c in ["click", "roi", "slice"]:
        for key, label in metrics:
            dlt = (w[f"{key}_{c}"]["Followup"] - w[f"{key}_{c}"]["Baseline"]).loc[keep]
            rr, pp = stats.pearsonr(expo.loc[keep], dlt); rrows.append(dict(convention=c, outcome=label, n=len(keep), pearson_r=rr, p=pp))
    rdf = pd.DataFrame(rrows); rdf.to_csv(OUT / "sensitivity_per_slice_rapa.csv", index=False, float_format="%.4f")
    md += ["", "## Rapamycin concentration vs Δ (n = 11)", "", "| Convention | Outcome | r | p |", "|---|---|---|---|"]
    for _, r in rdf.iterrows():
        md.append(f"| {r.convention} | {r.outcome} | {r.pearson_r:+.2f} | {r.p:.3f} |")
    (OUT / "sensitivity_per_slice_table1.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
