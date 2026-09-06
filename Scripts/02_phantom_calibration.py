#!/usr/bin/env python3
"""
02_phantom_calibration.py - Phantom calibration loader

All 26 scans in this study were calibrated with manual_calibration.py (one
click per rod on one slice; 4 mm-radius cylinders over 9 slices). The
automatic Hough-circle detector that used to live in this file was never used
for any reported number and has been removed (adversarial review 2026-09,
findings F07 and L03).

This module now only:
1. loads the manual calibration files written by manual_calibration.py
   (phantom_calibration.json, calibration_bmd.json, calibration_hu_stability.json)
2. exposes load_calibration() for callers that need slope/intercept/offset

The per-vertebra co-located calibration is derived downstream from
phantom_zprofile.json (02_phantom_zprofile.py, utils/calibration.py).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PhantomRod:
    name: str
    known_density_mgcm3: float
    center_voxel: Tuple[int, int, int]
    radius_voxels: float
    measured_hu_mean: float
    measured_hu_std: float
    voxel_count: int


@dataclass
class CalibrationResult:
    success: bool
    slope: float
    intercept: float
    r_squared: float
    drift_offset_hu: float
    rods: List[PhantomRod]
    qc_passed: bool
    qc_messages: List[str]


def load_manual_calibration(output_dir: Path) -> Optional[CalibrationResult]:
    """Load the manual calibration written by manual_calibration.py, if present."""
    manual_path = Path(output_dir) / "phantom_calibration.json"
    if not manual_path.exists():
        return None
    with open(manual_path) as f:
        data = json.load(f)
    if data.get("method") != "manual_clicker":
        logger.warning("phantom_calibration.json is not from manual_clicker")
        return None
    cal = data.get("calibration", {})
    rods = [PhantomRod(name=r["name"], known_density_mgcm3=r["density_mgcm3"],
                       center_voxel=tuple(r.get("center_voxel", [0, 0, 0])),
                       radius_voxels=data.get("parameters", {}).get("sample_radius_mm", 4.0),
                       measured_hu_mean=r["mean_hu"], measured_hu_std=r.get("std_hu", 0.0),
                       voxel_count=r.get("voxel_count", 0))
            for r in data.get("rods", [])]
    qc = data.get("qc", {})
    return CalibrationResult(success=True, slope=cal.get("slope", 0.0), intercept=cal.get("intercept", 0.0),
                             r_squared=cal.get("r_squared", 0.0),
                             drift_offset_hu=cal.get("drift_correction_hu", 0.0), rods=rods,
                             qc_passed=qc.get("passed", False),
                             qc_messages=qc.get("messages", ["Loaded from manual calibration"]))


def calibrate_phantom(ct_path: Path, output_dir: Path, **_ignored) -> CalibrationResult:
    """
    Return the manual calibration for this session. There is no automatic
    fallback: if the manual files are missing, run manual_calibration.py.
    """
    result = load_manual_calibration(output_dir)
    if result is not None:
        logger.info(f"Using manual calibration from {output_dir}")
        return result
    msg = (f"No manual calibration in {output_dir}. Run "
           f"'python Scripts/manual_calibration.py --data <RawData/bmd_ct> --output <project root>' "
           f"and then 'python Scripts/02_phantom_zprofile.py ...' for this session.")
    logger.error(msg)
    return CalibrationResult(success=False, slope=0.0, intercept=0.0, r_squared=0.0, drift_offset_hu=0.0,
                             rods=[], qc_passed=False, qc_messages=[msg])


def load_calibration(calibration_dir: Path) -> Tuple[float, float, float]:
    """Click-slice slope, intercept and drift offset from the saved JSON files."""
    calibration_dir = Path(calibration_dir)
    with open(calibration_dir / "calibration_bmd.json") as f:
        bmd_cal = json.load(f)
    with open(calibration_dir / "calibration_hu_stability.json") as f:
        stability_cal = json.load(f)
    return (bmd_cal["regression"]["slope"], bmd_cal["regression"]["intercept"],
            stability_cal["drift_correction"]["offset_hu"])


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print("Usage: python 02_phantom_calibration.py <ct_path> <session_derived_dir>")
        sys.exit(1)
    r = calibrate_phantom(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Calibration {'loaded' if r.success else 'MISSING'}: slope {r.slope:.4f}, intercept {r.intercept:.2f}, "
          f"R² {r.r_squared:.4f}, drift offset {r.drift_offset_hu:+.2f} HU")
