"""
Co-located phantom calibration helpers.

Added 2026-09 after the adversarial review (finding F02). The manual clicks in
phantom_calibration.json sample the rods on 9 slices around the click slice
(z0, typically 50). Rod HU varies systematically along z (beam hardening from
the vertebral bodies; the 200 rod reads up to ~35 HU lower at body levels than
at disc levels), so one line fitted at z0 is not representative of a vertebra
centred 40 mm away. 02_phantom_zprofile.py follows the tray slice by slice from
the clicks and stores per-slice rod means in phantom_zprofile.json. The helpers
here average those per-slice values over the slices of a given ROI and fit the
calibration line there (Cann-Genant "same slices as the vertebra" practice).
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from scipy import stats

REG_RODS = ["base", "bone_50", "bone_100", "bone_200"]
REG_DENSITIES = [0.0, 50.0, 100.0, 200.0]
MIN_SLICES = 5


def load_zprofile(derived_dir: Path) -> Optional[dict]:
    p = Path(derived_dir) / "phantom_zprofile.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _rod_arrays(zprofile: dict) -> Dict[str, np.ndarray]:
    ps = zprofile["per_slice"]
    out = {}
    for k in REG_RODS + ["fat"]:
        out[k] = np.array([np.nan if v is None else float(v) for v in ps[f"{k}_mean_hu"]], dtype=float)
    return out


def _valid_slices(rods: Dict[str, np.ndarray], z_indices: Iterable[int]) -> np.ndarray:
    z = np.array(sorted({int(i) for i in z_indices}), dtype=int)
    if z.size == 0:
        return z
    z = z[(z >= 0) & (z < rods["base"].size)]
    ok = np.all([np.isfinite(rods[k][z]) for k in REG_RODS], axis=0)
    return z[ok]


def calibration_for_slices(zprofile: dict, z_indices: Iterable[int]) -> dict:
    """
    Fit density = slope * HU + intercept from the four regression rods, each
    averaged over the given slices (tracked-circle per-slice means).
    """
    rods = _rod_arrays(zprofile)
    valid = _valid_slices(rods, z_indices)
    if valid.size < MIN_SLICES:
        raise ValueError(f"only {valid.size} usable phantom slices for calibration (need {MIN_SLICES})")
    means = {k: float(np.mean(rods[k][valid])) for k in REG_RODS}
    fit = stats.linregress([means[k] for k in REG_RODS], REG_DENSITIES)
    return {
        "method": "co-located: tracked rod means averaged over the ROI slices",
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r_squared": float(fit.rvalue ** 2),
        "rod_mean_hu": means,
        "fat_rod_mean_hu": float(np.nanmean(rods["fat"][valid])),
        "z_min": int(valid.min()),
        "z_max": int(valid.max()),
        "n_slices": int(valid.size),
    }


def base_offset_for_slices(zprofile: dict, z_indices: Iterable[int]) -> dict:
    """
    Scanner drift offset (HU to ADD to raw HU) from the water-equivalent base
    material averaged over the given slices.
    """
    rods = _rod_arrays(zprofile)
    valid = _valid_slices(rods, z_indices)
    if valid.size < MIN_SLICES:
        raise ValueError(f"only {valid.size} usable phantom slices for drift offset (need {MIN_SLICES})")
    base_mean = float(np.mean(rods["base"][valid]))
    return {
        "method": "co-located: tracked base-material mean over the ROI slices",
        "base_mean_hu": base_mean,
        "offset_hu": -base_mean,
        "z_min": int(valid.min()),
        "z_max": int(valid.max()),
        "n_slices": int(valid.size),
    }


def z50_calibration(calibration_dir: Path) -> dict:
    """The original click-slice calibration (calibration_bmd.json), for reference."""
    with open(Path(calibration_dir) / "calibration_bmd.json") as f:
        cal = json.load(f)
    reg = cal["regression"]
    return {"method": "click slice (9 slices around z0)", "slope": float(reg["slope"]),
            "intercept": float(reg["intercept"]), "r_squared": float(reg.get("r_squared", np.nan)),
            "z_range": cal.get("phantom_z_range")}
