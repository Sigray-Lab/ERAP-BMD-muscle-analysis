#!/usr/bin/env python3
"""
Step 1c: Build body-only L1/L2 masks (v2) from the vertebrae_body segmentation
and produce side-by-side QC against the current masks. ADDITIVE: writes to

    DerivedData/<sub>/<ses>/vertebral_bodies_v2/
        L1_body.nii.gz, L2_body.nii.gz          (endplates excluded, body only)
        trabecular_masks/L1_trabecular.nii.gz   (5 mm erosion, same as pipeline)
        body_isolation.json                     (instances, assignment, warnings)
    QC/<sub>/<ses>/body_isolation_v2.png
    QC/body_isolation_v2_summary.csv, QC/body_isolation_v2_sagittal_*.png

and a provisional vBMD per vertebra using the CURRENT (z=50) calibration, so
that the effect of the mask change alone can be seen. Nothing in
vertebral_bodies/, bone_results.json or Outputs/ is touched.

Which two vertebrae are used is taken from the existing
vertebral_bodies/vertebra_detection.json (so the sub-114 Z-split override stays
in force). Only the shape of each body changes.

Usage:
    python Scripts/01c_isolate_bodies_v2.py --data ../RawData/bmd_ct --output .
"""

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.body_isolation import isolate_bodies  # noqa: E402
bone_analysis = importlib.import_module("03_bone_analysis")

SESSIONS = ["ses-Baseline", "ses-Followup"]


def find_ct(data_dir: Path, sub: str, ses: str) -> Path:
    c = sorted((data_dir / sub / ses / "ct").glob("*_rec-stnd1.25mm_ct.nii.gz"))
    if len(c) != 1:
        raise FileNotFoundError(f"{sub}/{ses}: expected one 1.25 mm CT, found {len(c)}")
    return c[0]


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.count_nonzero(a & b) / s) if s else np.nan


def bmd_of(ct, core, slope, intercept):
    hu = ct[core]
    hu = hu[(hu >= bone_analysis.HU_MIN) & (hu <= bone_analysis.HU_MAX)]
    return float(slope * hu.mean() + intercept) if hu.size else np.nan, int(hu.size)


def process(sub, ses, data_dir: Path, root: Path):
    derived = root / "DerivedData" / sub / ses
    qc_dir = root / "QC" / sub / ses
    body_path = derived / "segmentations" / "vertebrae_body" / "vertebrae_body.nii.gz"
    det_path = derived / "vertebral_bodies" / "vertebra_detection.json"
    if not body_path.exists() or not det_path.exists():
        print(f"{sub}/{ses}: missing inputs, skipping")
        return None
    ct_path = find_ct(data_dir, sub, ses)
    nii = nib.load(ct_path)
    ct = np.asarray(nii.dataobj, dtype=np.float32)
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    body = np.asarray(nib.load(body_path).dataobj) > 0

    det = json.load(open(det_path))
    label_masks = {}
    for lvl, key in [("L1", "l1"), ("L2", "l2")]:
        p = root / det[key]["nifti_path"]       # stored project-relative
        if not p.exists():
            p = derived / "segmentations" / Path(det[key]["nifti_path"]).name
        label_masks[lvl] = np.asarray(nib.load(p).dataobj) > 0

    bodies, res = isolate_bodies(body, label_masks, spacing)
    out = derived / "vertebral_bodies_v2"
    (out / "trabecular_masks").mkdir(parents=True, exist_ok=True)
    rec = {"timestamp": datetime.now().isoformat(), "source_body_mask": str(body_path.relative_to(root)),
           "selected_labels": {lvl: det[k]["original_label"] for lvl, k in [("L1", "l1"), ("L2", "l2")]},
           "result": res.to_dict()}
    if not res.success:
        json.dump(rec, open(out / "body_isolation.json", "w"), indent=1)
        print(f"{sub}/{ses}: FAILED {res.warnings}")
        return {"sub": sub, "ses": ses, "success": False, "warnings": "; ".join(res.warnings)}

    cal = json.load(open(derived / "calibration_bmd.json"))["regression"]
    bj = json.load(open(derived / "bone_results.json"))
    row = {"sub": sub, "ses": ses, "success": True, "n_splits": len(res.splits),
           "warnings": "; ".join(res.warnings)}
    old = {}
    new_core = {}
    for lvl in ["L1", "L2"]:
        img = nib.Nifti1Image(bodies[lvl].astype(np.uint8), nii.affine)
        nib.save(img, out / f"{lvl}_body.nii.gz")
        core_img = bone_analysis.get_trabecular_mask(img)
        nib.save(core_img, out / "trabecular_masks" / f"{lvl}_trabecular.nii.gz")
        core = np.asarray(core_img.dataobj) > 0
        new_core[lvl] = core
        old[lvl] = {"body": np.asarray(nib.load(derived / "vertebral_bodies" / f"{lvl}_body.nii.gz").dataobj) > 0,
                    "core": np.asarray(nib.load(derived / "vertebral_bodies" / "trabecular_masks" / f"{lvl}_trabecular.nii.gz").dataobj) > 0}
        bmd_new, n_new = bmd_of(ct, core, cal["slope"], cal["intercept"])
        vox = float(np.prod(spacing)) / 1000
        row.update({
            f"{lvl}_label": det["l1" if lvl == "L1" else "l2"]["original_label"],
            f"{lvl}_body_z": f"{res.body_z_range[lvl][0]}-{res.body_z_range[lvl][1]}",
            f"{lvl}_body_old_cm3": round(old[lvl]["body"].sum() * vox, 2),
            f"{lvl}_body_new_cm3": round(bodies[lvl].sum() * vox, 2),
            f"{lvl}_body_dice_old_new": round(dice(old[lvl]["body"], bodies[lvl]), 3),
            f"{lvl}_core_old_cm3": round(bj[lvl]["trabecular_volume_cm3"], 2),
            f"{lvl}_core_new_cm3": round(n_new * vox, 2),
            f"{lvl}_old_core_outside_new_body_pct": round(100 * np.count_nonzero(old[lvl]["core"] & ~bodies[lvl]) / max(old[lvl]["core"].sum(), 1), 2),
            f"{lvl}_bmd_old": round(bj[lvl]["vBMD_mean_mgcm3"], 2),
            f"{lvl}_bmd_new_same_calib": round(bmd_new, 2),
            f"{lvl}_bmd_delta": round(bmd_new - bj[lvl]["vBMD_mean_mgcm3"], 2),
        })
    rec["provisional_bmd_current_calibration"] = {lvl: row[f"{lvl}_bmd_new_same_calib"] for lvl in ["L1", "L2"]}
    json.dump(rec, open(out / "body_isolation.json", "w"), indent=1)

    # ---------------- QC figure ----------------
    qc_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    both = bodies["L1"] | bodies["L2"] | old["L1"]["body"] | old["L2"]["body"]
    idx = np.where(both)
    x0, x1 = idx[0].min() - 12, idx[0].max() + 12
    y0, y1 = idx[1].min() - 12, idx[1].max() + 12
    xs = int(np.median(np.where(bodies["L1"] | bodies["L2"])[0]))
    for i, lvl in enumerate(["L1", "L2"]):
        z = int(np.median(np.where(new_core[lvl])[2]))
        for j, (title, bmask, cmask, cols) in enumerate([
                ("current (whole-label largest component)", old[lvl]["body"], old[lvl]["core"], ("orange", "red")),
                ("v2 (vertebrae_body instance)", bodies[lvl], new_core[lvl], ("lime", "cyan"))]):
            ax = axs[i, j]
            ax.imshow(ct[:, :, z].T, cmap="gray", vmin=-150, vmax=550, origin="lower")
            ax.contour(bmask[:, :, z].T, levels=[.5], colors=[cols[0]], linewidths=1.0)
            if cmask[:, :, z].any():
                ax.contour(cmask[:, :, z].T, levels=[.5], colors=[cols[1]], linewidths=1.0)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.set_title(f"{lvl} {title}, z={z}\nbody {row[f'{lvl}_body_old_cm3' if j == 0 else f'{lvl}_body_new_cm3']} cm3, "
                         f"core {row[f'{lvl}_core_old_cm3' if j == 0 else f'{lvl}_core_new_cm3']} cm3, "
                         f"vBMD {row[f'{lvl}_bmd_old' if j == 0 else f'{lvl}_bmd_new_same_calib']:.1f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        ax = axs[i, 2]
        ax.imshow(ct[xs, :, :].T, cmap="gray", vmin=-150, vmax=550, origin="lower", aspect=spacing[2] / spacing[1])
        ax.contour(old[lvl]["body"][xs].T, levels=[.5], colors=["orange"], linewidths=1.0)
        ax.contour(bodies[lvl][xs].T, levels=[.5], colors=["lime"], linewidths=1.0)
        ax.contour(old[lvl]["core"][xs].T, levels=[.5], colors=["red"], linewidths=.8)
        ax.contour(new_core[lvl][xs].T, levels=[.5], colors=["cyan"], linewidths=.8)
        ax.set_xlim(y0, y1)
        ax.set_title(f"{lvl} sagittal x={xs}: current orange/red, v2 green/cyan\nbody z {row[f'{lvl}_body_z']}, "
                     f"Dice old/new body {row[f'{lvl}_body_dice_old_new']}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    w = "; ".join(res.warnings) if res.warnings else "none"
    fig.suptitle(f"{sub} {ses}   labels: L1={row['L1_label']}, L2={row['L2_label']}   splits: {len(res.splits)}   warnings: {w}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(qc_dir / "body_isolation_v2.png", dpi=100)
    plt.close(fig)
    print(f"{sub}/{ses}: L1 {row['L1_bmd_old']:.1f}->{row['L1_bmd_new_same_calib']:.1f}  "
          f"L2 {row['L2_bmd_old']:.1f}->{row['L2_bmd_new_same_calib']:.1f}  "
          f"cores L1 {row['L1_core_old_cm3']}->{row['L1_core_new_cm3']}  L2 {row['L2_core_old_cm3']}->{row['L2_core_new_cm3']}  "
          f"{'WARN: ' + w if res.warnings else ''}")
    # keep for the sagittal contact sheet
    row["_sag"] = (ct[xs, y0:y1, :].T, {lvl: (old[lvl]["body"][xs, y0:y1, :].T, bodies[lvl][xs, y0:y1, :].T,
                                              old[lvl]["core"][xs, y0:y1, :].T, new_core[lvl][xs, y0:y1, :].T) for lvl in ["L1", "L2"]},
                   spacing[2] / spacing[1])
    return row


def contact_sheets(rows, root: Path):
    rows = [r for r in rows if r.get("_sag")]
    per = 13
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        fig, axs = plt.subplots(1, len(chunk), figsize=(2.6 * len(chunk), 7))
        for ax, r in zip(np.atleast_1d(axs), chunk):
            img, masks, asp = r["_sag"]
            ax.imshow(img, cmap="gray", vmin=-150, vmax=550, origin="lower", aspect=asp)
            for lvl in ["L1", "L2"]:
                ob, nb, oc, nc = masks[lvl]
                ax.contour(ob, levels=[.5], colors=["orange"], linewidths=.7)
                ax.contour(nb, levels=[.5], colors=["lime"], linewidths=.7)
                ax.contour(oc, levels=[.5], colors=["red"], linewidths=.6)
                ax.contour(nc, levels=[.5], colors=["cyan"], linewidths=.6)
            ax.set_title(f"{r['sub']}\n{r['ses'][4:]}" + ("\nWARN" if r["warnings"] else ""), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Sagittal: current body orange / core red;  v2 body green / core cyan", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(root / "QC" / f"body_isolation_v2_sagittal_{s // per + 1}.png", dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("."))
    ap.add_argument("--subject")
    a = ap.parse_args()
    root = a.output.resolve(); data_dir = a.data.resolve()
    subjects = [a.subject] if a.subject else sorted(p.name for p in data_dir.glob("sub-*"))
    rows = []
    for sub in subjects:
        for ses in SESSIONS:
            if (data_dir / sub / ses).exists():
                r = process(sub, ses, data_dir, root)
                if r:
                    rows.append(r)
    if len(rows) > 1:
        contact_sheets(rows, root)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df.to_csv(root / "QC" / "body_isolation_v2_summary.csv", index=False)
    warn = df[df["warnings"].astype(str).str.len() > 0]
    print(f"\n{len(df)} sessions, {len(warn)} with warnings:")
    for _, r in warn.iterrows():
        print(f"  {r['sub']} {r['ses']}: {r['warnings']}")


if __name__ == "__main__":
    main()
