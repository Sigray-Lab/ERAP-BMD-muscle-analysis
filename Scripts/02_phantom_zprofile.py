#!/usr/bin/env python3
"""
Step 2b: Follow the phantom rods along z from the manual clicks, and produce
strict QC so a human can verify the sampling circle never leaves the rod.

Added 2026-09 after the adversarial review (finding F02). Background:
- manual_calibration.py stores one click per rod on one slice (z0, typically 50)
  and averages a 4 mm-radius cylinder over z0-4..z0+4 (9 slices).
- Rod HU varies systematically along z (beam hardening from the vertebral
  bodies: the 200 rod reads ~35 HU lower at body levels than at disc levels).
- The phantom tray may be slightly tilted, so a fixed circle could in principle
  drift toward the rod edge over 80 mm. Measured worst case in this cohort is
  2.8 mm; the rod is ~15 mm wide; the circle is 8 mm wide.

What this script does (ADDITIVE, changes no existing result):
1. Uses the whole phantom patch (all 5 rods) and phase correlation against the
   click slice to estimate the in-plane shift of the tray on every slice.
2. Samples every rod on every slice with the 4 mm circle at the shifted centre
   (per-slice mean, SD, and an inner-vs-annulus edge check).
3. Reproduces the original 9-slice fixed-centre value at z0 to prove the
   coordinate convention (must match phantom_calibration.json to <0.05 HU).
4. Computes, but does not yet use, per-vertebra calibration lines from the rod
   means averaged over the slices of each existing trabecular core.
5. Writes DerivedData/<sub>/<ses>/phantom_zprofile.json and
   QC/<sub>/<ses>/phantom_tracking.png, plus cohort contact sheets
   QC/phantom_tracking_extremes_*.png showing the circle at the first and last
   slice of every scan for every rod.

QC flags written to the JSON (any flag -> "needs_review": true):
    shift_gt_4mm            tray shift exceeds 4 mm on some slice
    tracking_lost           phase correlation shift > 12 voxels (slice marked NaN)
    sd_ratio_gt_2p5         per-slice SD > 2.5x the SD at z0 for a bone rod
    edge_gt_30hu            |inner - annulus| > 30 HU for a bone rod (circle at rod edge)
    order_violation         base < 50 < 100 < 200 not satisfied on some slice
    reproduction_failed     fixed-centre 9-slice mean does not match the stored value

Usage:
    python Scripts/02_phantom_zprofile.py --data ../RawData/bmd_ct --output .
    python Scripts/02_phantom_zprofile.py --data ../RawData/bmd_ct --output . --subject sub-109
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import stats
from skimage.registration import phase_cross_correlation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SESSIONS = ["ses-Baseline", "ses-Followup"]
ROD_ORDER = ["fat", "base", "bone_50", "bone_100", "bone_200"]
REG_RODS = ["base", "bone_50", "bone_100", "bone_200"]
SAMPLE_RADIUS_MM = 4.0
SLAB_HALF = 4                 # 9-slice slab for tracking (matches manual_calibration)
PATCH_MARGIN_VOX = 22
MAX_SHIFT_VOX = 12.0          # beyond this, tracking is declared lost for that slice
FLAG_SHIFT_MM = 4.0
FLAG_SD_RATIO = 2.5
FLAG_EDGE_HU = 30.0
CROP = 35                     # half-size of QC crops in voxels


# ----------------------------------------------------------------------------
def find_ct(data_dir: Path, sub: str, ses: str) -> Path:
    c = sorted((data_dir / sub / ses / "ct").glob("*_rec-stnd1.25mm_ct.nii.gz"))
    if len(c) != 1:
        raise FileNotFoundError(f"{sub}/{ses}: expected one 1.25 mm CT, found {len(c)}")
    return c[0]


def circle_mask(shape, c0, c1, r):
    i, j = np.ogrid[:shape[0], :shape[1]]
    return ((i - c0) ** 2 + (j - c1) ** 2) <= r ** 2


def annulus_mask(shape, c0, c1, r_in, r_out):
    i, j = np.ogrid[:shape[0], :shape[1]]
    d2 = (i - c0) ** 2 + (j - c1) ** 2
    return (d2 > r_in ** 2) & (d2 <= r_out ** 2)


def core_zrange(vb_dir: Path, level: str):
    p = vb_dir / "trabecular_masks" / f"{level}_trabecular.nii.gz"
    if not p.exists():
        return None
    z = np.where((nib.load(p).get_fdata() > 0).any(axis=(0, 1)))[0]
    return (int(z.min()), int(z.max())) if len(z) else None


def fit(rod_means: dict):
    hu = [rod_means[k] for k in REG_RODS]
    dens = [0, 50, 100, 200]
    r = stats.linregress(hu, dens)
    return {"slope": float(r.slope), "intercept": float(r.intercept),
            "r_squared": float(r.rvalue ** 2), "rod_mean_hu": {k: float(v) for k, v in rod_means.items()}}


# ----------------------------------------------------------------------------
def process_session(sub, ses, data_dir: Path, root: Path):
    derived = root / "DerivedData" / sub / ses
    qc_dir = root / "QC" / sub / ses
    qc_dir.mkdir(parents=True, exist_ok=True)
    cal_path = derived / "phantom_calibration.json"
    if not cal_path.exists():
        print(f"{sub}/{ses}: no phantom_calibration.json, skipping")
        return None
    cal = json.load(open(cal_path))
    ct_path = find_ct(data_dir, sub, ses)
    nii = nib.load(ct_path)
    ct = np.asarray(nii.dataobj, dtype=np.float32)
    zooms = nii.header.get_zooms()[:3]
    inplane = float(np.mean(zooms[:2]))
    nz = ct.shape[2]
    r_vox = SAMPLE_RADIUS_MM / inplane

    rods = {r["name"]: r for r in cal["rods"]}
    centres = {k: np.array(rods[k]["center_voxel"][:2], dtype=float) for k in ROD_ORDER}
    z0s = {rods[k]["center_voxel"][2] for k in ROD_ORDER}
    z0 = int(round(float(np.median(list(z0s)))))

    # --- phantom patch (all rods) --------------------------------------------
    pts = np.array(list(centres.values()))
    lo = np.maximum(np.floor(pts.min(axis=0)).astype(int) - PATCH_MARGIN_VOX, 0)
    hi = np.minimum(np.ceil(pts.max(axis=0)).astype(int) + PATCH_MARGIN_VOX + 1, ct.shape[:2])
    patch = ct[lo[0]:hi[0], lo[1]:hi[1], :]
    cs = np.concatenate([np.zeros(patch.shape[:2] + (1,), np.float64),
                         np.cumsum(patch.astype(np.float64), axis=2)], axis=2)

    def slab(z):
        a, b = max(0, z - SLAB_HALF), min(nz - 1, z + SLAB_HALF)
        return (cs[:, :, b + 1] - cs[:, :, a]) / (b - a + 1)

    ref = slab(z0)

    # --- track and sample -----------------------------------------------------
    shift = np.full((nz, 2), np.nan)
    mean = {k: np.full(nz, np.nan) for k in ROD_ORDER}
    sd = {k: np.full(nz, np.nan) for k in ROD_ORDER}
    edge = {k: np.full(nz, np.nan) for k in ROD_ORDER}
    for z in range(nz):
        s, _, _ = phase_cross_correlation(ref, slab(z), upsample_factor=4, normalization=None)
        if np.linalg.norm(s) > MAX_SHIFT_VOX:
            continue
        shift[z] = s
        sl = ct[:, :, z]
        for k, c in centres.items():
            c0, c1 = c[0] - s[0], c[1] - s[1]
            m = circle_mask(sl.shape, c0, c1, r_vox)
            a = annulus_mask(sl.shape, c0, c1, r_vox, r_vox + 2.0)
            v = sl[m]
            mean[k][z], sd[k][z] = float(v.mean()), float(v.std())
            edge[k][z] = float(v.mean() - sl[a].mean())

    tracked_centre = {k: np.stack([centres[k][0] - shift[:, 0], centres[k][1] - shift[:, 1]], axis=1)
                      for k in ROD_ORDER}

    # --- reproduction check of the stored 9-slice fixed-centre value ---------
    repro = {}
    for k, c in centres.items():
        m = circle_mask(ct.shape[:2], c[0], c[1], r_vox)
        vals = ct[:, :, z0 - SLAB_HALF:z0 + SLAB_HALF + 1][m]
        repro[k] = {"recomputed": float(vals.mean()), "stored": float(rods[k]["mean_hu"]),
                    "n_recomputed": int(vals.size), "n_stored": int(rods[k]["voxel_count"])}
    repro_ok = all(abs(v["recomputed"] - v["stored"]) < 0.05 and v["n_recomputed"] == v["n_stored"]
                   for v in repro.values())

    # --- flags ----------------------------------------------------------------
    shift_mm = np.linalg.norm(shift, axis=1) * inplane
    lost = np.isnan(shift[:, 0])
    flags = {
        "shift_gt_4mm": bool(np.nanmax(shift_mm) > FLAG_SHIFT_MM),
        "tracking_lost": bool(lost.any()),
        "sd_ratio_gt_2p5": bool(any(np.nanmax(sd[k] / sd[k][z0]) > FLAG_SD_RATIO for k in REG_RODS)),
        "edge_gt_30hu": bool(any(np.nanmax(np.abs(edge[k])) > FLAG_EDGE_HU for k in ["bone_50", "bone_100", "bone_200"])),
        "order_violation": bool(np.any((mean["base"] >= mean["bone_50"]) | (mean["bone_50"] >= mean["bone_100"])
                                       | (mean["bone_100"] >= mean["bone_200"]))),
        "reproduction_failed": not repro_ok,
    }

    # --- per-vertebra calibration (computed, not yet consumed) --------------
    vb_dir = derived / "vertebral_bodies"
    cores = {lvl: core_zrange(vb_dir, lvl) for lvl in ["L1", "L2"]}
    calib = {"global_z50_stored": {"slope": cal["calibration"]["slope"],
                                   "intercept": cal["calibration"]["intercept"],
                                   "r_squared": cal["calibration"]["r_squared"],
                                   "slices": [z0 - SLAB_HALF, z0 + SLAB_HALF]}}
    valid = ~lost
    calib["all_slices_tracked"] = fit({k: np.nanmean(mean[k][valid]) for k in ROD_ORDER})
    calib["all_slices_tracked"]["slices"] = [int(np.where(valid)[0].min()), int(np.where(valid)[0].max())]
    for lvl, zr in cores.items():
        if zr is None:
            continue
        sel = valid.copy(); sel[:zr[0]] = False; sel[zr[1] + 1:] = False
        calib[f"{lvl}_core_tracked"] = fit({k: np.nanmean(mean[k][sel]) for k in ROD_ORDER})
        calib[f"{lvl}_core_tracked"]["slices"] = list(zr)
        calib[f"{lvl}_core_tracked"]["n_slices"] = int(sel.sum())
    calib["muscle_slab_base_offset_hu"] = None
    if all(cores.values()):
        zmin = min(c[0] for c in cores.values()); zmax = max(c[1] for c in cores.values())
        sel = valid.copy(); sel[:zmin] = False; sel[zmax + 1:] = False
        calib["muscle_slab_base_offset_hu"] = float(-np.nanmean(mean["base"][sel]))

    out = {
        "method": "phase-correlation tray tracking from manual clicks; per-slice 4 mm circle sampling",
        "timestamp": datetime.now().isoformat(),
        "source_calibration": str(cal_path.name),
        "ct": ct_path.name, "voxel_size_mm": [float(v) for v in zooms],
        "click_slice_z0": z0, "sample_radius_mm": SAMPLE_RADIUS_MM, "sample_radius_vox": float(r_vox),
        "patch_bbox": [int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])],
        "max_shift_mm": float(np.nanmax(shift_mm)),
        "flags": flags, "needs_review": any(flags.values()),
        "reproduction_of_stored_9slice_means": repro,
        "per_slice": {
            "shift_vox_axis0": [None if np.isnan(v) else round(float(v), 3) for v in shift[:, 0]],
            "shift_vox_axis1": [None if np.isnan(v) else round(float(v), 3) for v in shift[:, 1]],
            **{f"{k}_mean_hu": [None if np.isnan(v) else round(float(v), 2) for v in mean[k]] for k in ROD_ORDER},
            **{f"{k}_sd_hu": [None if np.isnan(v) else round(float(v), 2) for v in sd[k]] for k in ROD_ORDER},
            **{f"{k}_edge_hu": [None if np.isnan(v) else round(float(v), 2) for v in edge[k]] for k in ROD_ORDER},
        },
        "cores_used_for_calibration": cores,
        "calibration": calib,
    }
    json.dump(out, open(derived / "phantom_zprofile.json", "w"), indent=1)

    # --- QC figure per session -------------------------------------------------
    cols = [("first slice", 0)]
    if cores["L2"]:
        cols.append(("L2 core centre", int(np.mean(cores["L2"]))))
    cols.append((f"click slice", z0))
    if cores["L1"]:
        cols.append(("L1 core centre", int(np.mean(cores["L1"]))))
    cols.append(("last slice", nz - 1))
    th = np.linspace(0, 2 * np.pi, 120)
    fig = plt.figure(figsize=(3.0 * len(cols), 3.0 * len(ROD_ORDER) + 6.5))
    gs = fig.add_gridspec(len(ROD_ORDER) + 2, len(cols), height_ratios=[1] * len(ROD_ORDER) + [1.3, 1.3])
    for i, k in enumerate(ROD_ORDER):
        c = centres[k]
        for j, (name, z) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            a0 = int(max(0, c[0] - CROP)); a1 = int(max(0, c[1] - CROP))
            crop = ct[a0:a0 + 2 * CROP, a1:a1 + 2 * CROP, z].T
            ax.imshow(crop, cmap="gray", vmin=-250, vmax=450, origin="lower",
                      extent=[a0, a0 + crop.shape[1], a1, a1 + crop.shape[0]])
            ax.plot(c[0] + r_vox * np.cos(th), c[1] + r_vox * np.sin(th), "r-", lw=1.0, alpha=.6)
            if not lost[z]:
                t = tracked_centre[k][z]
                ax.plot(t[0] + r_vox * np.cos(th), t[1] + r_vox * np.sin(th), "c--", lw=1.6)
                ttl = f"{k} z={z}\n{mean[k][z]:.0f} HU, SD {sd[k][z]:.0f}, edge {edge[k][z]:+.0f}"
            else:
                ttl = f"{k} z={z}\nTRACKING LOST"
            ax.set_title(ttl, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.text(0.5, 1.28, name, transform=ax.transAxes, ha="center", fontsize=10, weight="bold")
    # drift
    ax = fig.add_subplot(gs[len(ROD_ORDER), 0:2])
    ax.plot(-shift[:, 0] * inplane, label="axis0 (mm)"); ax.plot(-shift[:, 1] * inplane, label="axis1 (mm)")
    ax.plot(shift_mm, "k", lw=.8, label="|shift| (mm)")
    ax.axhline(FLAG_SHIFT_MM, color="r", ls=":", lw=.8); ax.axvspan(z0 - SLAB_HALF, z0 + SLAB_HALF, color="gray", alpha=.25)
    ax.set(title=f"Tray shift vs click slice (max {np.nanmax(shift_mm):.2f} mm)", xlabel="slice"); ax.legend(fontsize=7)
    # HU profile
    ax = fig.add_subplot(gs[len(ROD_ORDER), 2:])
    for k in ROD_ORDER:
        ax.plot(mean[k], label=k)
    ax.axvspan(z0 - SLAB_HALF, z0 + SLAB_HALF, color="gray", alpha=.25, label="current 9 slices")
    for lvl, zr in cores.items():
        if zr:
            ax.axvspan(zr[0], zr[1], color="orange", alpha=.12); ax.text(np.mean(zr), ax.get_ylim()[1] * .95, f"{lvl} core", ha="center", fontsize=8)
    ax.set(title="Rod mean HU per slice (tracked circle)", xlabel="slice"); ax.legend(fontsize=7, ncol=3)
    # SD / edge
    ax = fig.add_subplot(gs[len(ROD_ORDER) + 1, 0:2])
    for k in REG_RODS:
        ax.plot(sd[k] / sd[k][z0], label=k)
    ax.axhline(FLAG_SD_RATIO, color="r", ls=":", lw=.8); ax.set(title="Within-circle SD relative to click slice", xlabel="slice"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[len(ROD_ORDER) + 1, 2:])
    for k in ["bone_50", "bone_100", "bone_200"]:
        ax.plot(edge[k], label=k)
    ax.axhline(FLAG_EDGE_HU, color="r", ls=":", lw=.8); ax.axhline(-FLAG_EDGE_HU, color="r", ls=":", lw=.8)
    ax.set(title="Edge check: inner mean minus annulus mean (HU)", xlabel="slice"); ax.legend(fontsize=7)
    fl = ", ".join(k for k, v in flags.items() if v) or "none"
    fig.suptitle(f"{sub} {ses}  |  red = fixed click circle, cyan = tracked circle  |  flags: {fl}"
                 f"  |  stored-value reproduction: {'OK' if repro_ok else 'FAILED'}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(qc_dir / "phantom_tracking.png", dpi=100)
    plt.close(fig)

    print(f"{sub}/{ses}: max shift {np.nanmax(shift_mm):.2f} mm, repro {'OK' if repro_ok else 'FAIL'}, "
          f"flags: {fl}")
    return {"sub": sub, "ses": ses, "ct": ct, "centres": centres, "tracked": tracked_centre,
            "lost": lost, "r_vox": r_vox, "nz": nz, "mean": mean, "flags": flags,
            "max_shift_mm": float(np.nanmax(shift_mm))}


# ----------------------------------------------------------------------------
def contact_sheets(results, root: Path):
    """Extreme-end sheets: every session x every regression rod at first and last slice."""
    rods = ["base", "bone_50", "bone_100", "bone_200"]
    th = np.linspace(0, 2 * np.pi, 120)
    per_sheet = 13
    for s in range(0, len(results), per_sheet):
        chunk = results[s:s + per_sheet]
        fig, axs = plt.subplots(len(chunk), 2 * len(rods), figsize=(2.0 * 2 * len(rods), 2.1 * len(chunk)))
        axs = np.atleast_2d(axs)
        for i, r in enumerate(chunk):
            for jr, k in enumerate(rods):
                for je, z in enumerate([0, r["nz"] - 1]):
                    ax = axs[i, 2 * jr + je]
                    c = r["centres"][k]
                    a0 = int(max(0, c[0] - 25)); a1 = int(max(0, c[1] - 25))
                    crop = r["ct"][a0:a0 + 50, a1:a1 + 50, z].T
                    ax.imshow(crop, cmap="gray", vmin=-250, vmax=450, origin="lower",
                              extent=[a0, a0 + crop.shape[1], a1, a1 + crop.shape[0]])
                    ax.plot(c[0] + r["r_vox"] * np.cos(th), c[1] + r["r_vox"] * np.sin(th), "r-", lw=.7, alpha=.5)
                    if not r["lost"][z]:
                        t = r["tracked"][k][z]
                        ax.plot(t[0] + r["r_vox"] * np.cos(th), t[1] + r["r_vox"] * np.sin(th), "c--", lw=1.4)
                    ax.set_xticks([]); ax.set_yticks([])
                    if i == 0:
                        ax.set_title(f"{k}\n{'first' if z == 0 else 'last'} slice", fontsize=8)
                    if 2 * jr + je == 0:
                        ax.set_ylabel(f"{r['sub']}\n{r['ses'].replace('ses-', '')}\n{r['max_shift_mm']:.1f} mm", fontsize=8)
        fig.suptitle("Tracked circle (cyan) at the extreme ends of every scan; red = fixed click circle", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(root / "QC" / f"phantom_tracking_extremes_{s // per_sheet + 1}.png", dpi=100)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("."))
    ap.add_argument("--subject")
    a = ap.parse_args()
    root = a.output.resolve(); data_dir = a.data.resolve()
    subjects = [a.subject] if a.subject else sorted(p.name for p in data_dir.glob("sub-*"))
    results = []
    for sub in subjects:
        for ses in SESSIONS:
            if (data_dir / sub / ses).exists():
                r = process_session(sub, ses, data_dir, root)
                if r:
                    results.append(r)
    if len(results) > 1:
        contact_sheets(results, root)
    summary = [{"sub": r["sub"], "ses": r["ses"], "max_shift_mm": round(r["max_shift_mm"], 2),
                "flags": [k for k, v in r["flags"].items() if v]} for r in results]
    json.dump(summary, open(root / "QC" / "phantom_tracking_summary.json", "w"), indent=1)
    flagged = [s for s in summary if s["flags"]]
    print(f"\n{len(results)} sessions processed, {len(flagged)} flagged for review")
    for s in flagged:
        print("  ", s)


if __name__ == "__main__":
    main()
