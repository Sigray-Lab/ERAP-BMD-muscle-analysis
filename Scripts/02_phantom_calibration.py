#!/usr/bin/env python3
"""
phantom_calibration.py - Density phantom detection and HU→BMD calibration

This module handles:
1. Detecting the calibration phantom in CT images (table-side restricted)
2. Identifying the 5 cylindrical rods and base material
3. Creating HU to BMD (mg/cm³) calibration curve
4. Generating calibration JSON files for bone and muscle analysis

Key improvements over basic Hough detection:
- Restricts search to anterior (table-side) region where phantom is located
- Uses HU-based filtering to identify phantom-like structures
- Requires row-like arrangement of detected circles
- Provides manual override capability for failed detections
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import nibabel as nib
import cv2
from scipy import stats
from scipy.ndimage import label

logger = logging.getLogger(__name__)


@dataclass
class PhantomRod:
    """Data for a single phantom rod."""
    name: str
    known_density_mgcm3: float
    center_voxel: Tuple[int, int, int]
    radius_voxels: float
    measured_hu_mean: float
    measured_hu_std: float
    voxel_count: int


@dataclass
class CalibrationResult:
    """Results from phantom calibration."""
    success: bool
    slope: float
    intercept: float
    r_squared: float
    drift_offset_hu: float
    rods: List[PhantomRod]
    qc_passed: bool
    qc_messages: List[str]


# Known phantom values
PHANTOM_RODS = {
    "base": {"density": 0, "expected_hu_range": (-50, 50)},
    "fat": {"density": -100, "expected_hu_range": (-150, -50)},  # QC only
    "bone_50": {"density": 50, "expected_hu_range": (20, 100)},
    "bone_100": {"density": 100, "expected_hu_range": (80, 200)},
    "bone_200": {"density": 200, "expected_hu_range": (150, 350)},
}


def find_phantom_slices_simple(ct_data: np.ndarray, n_slices_to_check: int = 40) -> Tuple[int, int]:
    """
    Find z-range containing the phantom using a simple approach.

    Looks in the inferior portion of the volume where the phantom
    calibration pad is typically located.

    Args:
        ct_data: 3D CT volume
        n_slices_to_check: Number of inferior slices to check

    Returns:
        Tuple of (z_start, z_end)
    """
    n_slices = ct_data.shape[2]

    # The phantom is in the inferior portion of the scan
    # Check slices 0 to n_slices_to_check (or 1/3 of volume, whichever is smaller)
    z_search_end = min(n_slices_to_check, n_slices // 3)

    # Find slices with reasonable content (not all air)
    valid_slices = []
    for z in range(0, z_search_end):
        slice_data = ct_data[:, :, z]
        if np.percentile(slice_data, 75) > -500:  # Has tissue content
            valid_slices.append(z)

    if valid_slices:
        return min(valid_slices), max(valid_slices)

    # Fallback
    return 0, min(30, n_slices - 1)


def detect_phantom_circles(ct_slice: np.ndarray,
                           min_radius_mm: float = 5.0,
                           max_radius_mm: float = 18.0,
                           voxel_size_xy: float = 1.0) -> List[Tuple[int, int, int]]:
    """
    Detect circular structures (phantom rods) in a CT slice.

    Restricts search to anterior region (table-side) where phantom is located.
    Uses Hough circle detection with multiple parameter sets.

    Args:
        ct_slice: 2D CT slice
        min_radius_mm: Minimum rod radius in mm
        max_radius_mm: Maximum rod radius in mm
        voxel_size_xy: In-plane voxel size in mm

    Returns:
        List of (x, y, radius) tuples for detected circles
    """
    h, w = ct_slice.shape

    # Convert radius to voxels
    min_radius = int(min_radius_mm / voxel_size_xy)
    max_radius = int(max_radius_mm / voxel_size_xy)

    # Search only in anterior band (first 30% of image height)
    # This is where the table/phantom is located
    anterior_band_height = int(0.30 * h)
    anterior_region = ct_slice[0:anterior_band_height, :]

    # Normalize for OpenCV
    img_norm = np.clip(anterior_region, -200, 400)
    img_norm = ((img_norm + 200) / 600 * 255).astype(np.uint8)

    # Try multiple detection parameter sets
    all_circles = []

    for blur_size in [(5, 5), (7, 7), (9, 9)]:
        blurred = cv2.GaussianBlur(img_norm, blur_size, 2)

        for param2 in [20, 25, 30, 35]:
            for min_dist in [25, 35, 45]:
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=min_dist,
                    param1=50,
                    param2=param2,
                    minRadius=min_radius,
                    maxRadius=max_radius
                )

                if circles is not None:
                    for circle in circles[0]:
                        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                        all_circles.append((x, y, r))

    # Deduplicate
    unique_circles = []
    for c in all_circles:
        is_dup = False
        for uc in unique_circles:
            dist = np.sqrt((c[0] - uc[0])**2 + (c[1] - uc[1])**2)
            if dist < min_radius:
                is_dup = True
                break
        if not is_dup:
            unique_circles.append(c)

    # Also try posterior band (in case phantom orientation is flipped)
    posterior_start = h - anterior_band_height
    posterior_region = ct_slice[posterior_start:h, :]

    img_norm_post = np.clip(posterior_region, -200, 400)
    img_norm_post = ((img_norm_post + 200) / 600 * 255).astype(np.uint8)

    posterior_circles = []
    blurred = cv2.GaussianBlur(img_norm_post, (7, 7), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=35,
        param1=50, param2=25, minRadius=min_radius, maxRadius=max_radius
    )
    if circles is not None:
        for circle in circles[0]:
            x, y, r = int(circle[0]), int(circle[1]) + posterior_start, int(circle[2])
            posterior_circles.append((x, y, r))

    # Score both and return better one
    def score_circles(circles_list):
        if not circles_list:
            return -1000
        # Prefer 4-7 circles with similar y-coordinates (row-like)
        if len(circles_list) < 4:
            return -500
        y_coords = [c[1] for c in circles_list]
        y_spread = max(y_coords) - min(y_coords)
        count_score = 10 - abs(len(circles_list) - 5) * 2
        spread_score = max(0, 50 - y_spread)
        return count_score + spread_score

    anterior_score = score_circles(unique_circles)
    posterior_score = score_circles(posterior_circles)

    if anterior_score >= posterior_score:
        return unique_circles
    else:
        return posterior_circles


def sample_rod_hu(ct_data: np.ndarray,
                  center: Tuple[int, int, int],
                  radius: int,
                  z_range: Tuple[int, int]) -> Tuple[float, float, int]:
    """
    Sample HU values from a cylindrical ROI.

    Args:
        ct_data: 3D CT volume
        center: (x, y, z_center) of the rod
        radius: Radius in voxels
        z_range: (z_start, z_end) slice range

    Returns:
        Tuple of (mean_hu, std_hu, voxel_count)
    """
    x, y, z_center = center
    z_start, z_end = z_range

    # Create circular mask - use slightly smaller radius to avoid edge effects
    yy, xx = np.ogrid[:ct_data.shape[0], :ct_data.shape[1]]
    inner_radius = max(1, radius - 2)
    circle_mask = ((xx - x)**2 + (yy - y)**2) <= inner_radius**2

    # Collect HU values across z-range
    hu_values = []
    for z in range(z_start, z_end + 1):
        if 0 <= z < ct_data.shape[2]:
            values = ct_data[:, :, z][circle_mask]
            hu_values.extend(values)

    hu_values = np.array(hu_values)

    if len(hu_values) == 0:
        return 0.0, 0.0, 0

    return float(np.mean(hu_values)), float(np.std(hu_values)), len(hu_values)


def identify_rods_by_hu(circles: List[Tuple[int, int, int]],
                        ct_data: np.ndarray,
                        z_range: Tuple[int, int]) -> dict:
    """
    Identify which detected circle corresponds to which phantom rod based on HU values.

    Args:
        circles: List of (x, y, radius) from circle detection
        ct_data: 3D CT volume
        z_range: Slice range for sampling

    Returns:
        Dictionary mapping rod name to PhantomRod dataclass
    """
    if len(circles) < 4:
        logger.warning(f"Only {len(circles)} circles detected, need at least 4 for calibration")
        return {}

    # Sample HU from each circle
    rod_samples = []
    z_center = (z_range[0] + z_range[1]) // 2

    for x, y, r in circles:
        mean_hu, std_hu, count = sample_rod_hu(ct_data, (x, y, z_center), r, z_range)
        rod_samples.append({
            "center": (x, y, z_center),
            "radius": r,
            "mean_hu": mean_hu,
            "std_hu": std_hu,
            "count": count
        })

    # Sort by HU value (ascending)
    rod_samples.sort(key=lambda x: x["mean_hu"])

    # Assign identities based on expected HU ordering
    # fat < base < bone_50 < bone_100 < bone_200
    identified = {}

    if len(rod_samples) >= 5:
        rod_names = ["fat", "base", "bone_50", "bone_100", "bone_200"]
        for i, name in enumerate(rod_names):
            sample = rod_samples[i]
            identified[name] = PhantomRod(
                name=name,
                known_density_mgcm3=PHANTOM_RODS[name]["density"],
                center_voxel=sample["center"],
                radius_voxels=sample["radius"],
                measured_hu_mean=sample["mean_hu"],
                measured_hu_std=sample["std_hu"],
                voxel_count=sample["count"]
            )
    elif len(rod_samples) >= 4:
        # Without fat rod
        logger.info("Only 4 circles detected, identifying without fat rod")

        # Find the rod closest to 0 HU for base
        base_idx = min(range(len(rod_samples)),
                       key=lambda i: abs(rod_samples[i]["mean_hu"]))

        bone_samples = [s for i, s in enumerate(rod_samples) if i != base_idx]
        bone_samples.sort(key=lambda x: x["mean_hu"])

        sample = rod_samples[base_idx]
        identified["base"] = PhantomRod(
            name="base",
            known_density_mgcm3=0,
            center_voxel=sample["center"],
            radius_voxels=sample["radius"],
            measured_hu_mean=sample["mean_hu"],
            measured_hu_std=sample["std_hu"],
            voxel_count=sample["count"]
        )

        for i, name in enumerate(["bone_50", "bone_100", "bone_200"]):
            if i < len(bone_samples):
                sample = bone_samples[i]
                identified[name] = PhantomRod(
                    name=name,
                    known_density_mgcm3=PHANTOM_RODS[name]["density"],
                    center_voxel=sample["center"],
                    radius_voxels=sample["radius"],
                    measured_hu_mean=sample["mean_hu"],
                    measured_hu_std=sample["std_hu"],
                    voxel_count=sample["count"]
                )

    return identified


def compute_calibration(rods: dict) -> Tuple[float, float, float]:
    """
    Compute linear regression for HU → BMD calibration.

    Uses base (0 mg/cm³) and bone rods (50, 100, 200 mg/cm³).
    Fat rod is excluded from calibration (used for QC only).

    Args:
        rods: Dictionary of identified PhantomRod objects

    Returns:
        Tuple of (slope, intercept, r_squared)
    """
    calibration_rods = ["base", "bone_50", "bone_100", "bone_200"]

    hu_values = []
    bmd_values = []

    for name in calibration_rods:
        if name in rods:
            hu_values.append(rods[name].measured_hu_mean)
            bmd_values.append(rods[name].known_density_mgcm3)

    if len(hu_values) < 3:
        logger.error(f"Insufficient calibration points: {len(hu_values)}")
        return 0.0, 0.0, 0.0

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(hu_values, bmd_values)
    r_squared = r_value ** 2

    logger.info(f"Calibration: slope={slope:.4f}, intercept={intercept:.2f}, R²={r_squared:.4f}")

    return slope, intercept, r_squared


def run_qc_checks(rods: dict, slope: float, intercept: float, r_squared: float) -> Tuple[bool, List[str]]:
    """
    Run quality control checks on calibration.
    """
    messages = []
    passed = True

    # Check R²
    if r_squared < 0.95:
        messages.append(f"WARNING: Calibration R² = {r_squared:.4f} < 0.95")
        # Don't fail on R² alone - might still be usable
    elif r_squared < 0.98:
        messages.append(f"WARNING: R² = {r_squared:.4f} < 0.98 (acceptable but not ideal)")
    else:
        messages.append(f"OK: R² = {r_squared:.4f}")

    # Check monotonic ordering
    calibration_rods = ["base", "bone_50", "bone_100", "bone_200"]
    hu_order = [rods[name].measured_hu_mean for name in calibration_rods if name in rods]

    if hu_order != sorted(hu_order):
        messages.append("FAIL: HU values not monotonically increasing")
        passed = False
    else:
        messages.append("OK: HU values monotonically increasing")

    # Check per-rod residuals
    for name in calibration_rods:
        if name in rods:
            rod = rods[name]
            predicted_bmd = slope * rod.measured_hu_mean + intercept
            residual = abs(predicted_bmd - rod.known_density_mgcm3)
            residual_hu = residual / slope if slope != 0 else float('inf')

            if residual_hu > 10:
                messages.append(f"WARNING: {name} residual = {residual_hu:.1f} HU > 10 HU threshold")
            else:
                messages.append(f"OK: {name} residual = {residual_hu:.1f} HU")

    # Check drift (base should be near 0 HU)
    if "base" in rods:
        drift = rods["base"].measured_hu_mean
        if abs(drift) > 50:
            messages.append(f"FAIL: Drift = {drift:.1f} HU > 50 HU threshold")
            passed = False
        elif abs(drift) > 30:
            messages.append(f"WARNING: Drift = {drift:.1f} HU > 30 HU")
        else:
            messages.append(f"OK: Drift = {drift:.1f} HU")

    return passed, messages


def load_manual_calibration(output_dir: Path) -> Optional[CalibrationResult]:
    """
    Load manual calibration if it exists.

    The manual_calibration.py script creates phantom_calibration.json with
    manually clicked rod positions. This function loads that and converts
    it to a CalibrationResult.

    Args:
        output_dir: Directory containing phantom_calibration.json

    Returns:
        CalibrationResult if manual calibration exists, None otherwise
    """
    manual_path = output_dir / "phantom_calibration.json"

    if not manual_path.exists():
        return None

    logger.info(f"Loading manual calibration from {manual_path}")

    with open(manual_path) as f:
        data = json.load(f)

    # Check if this is a manual calibration file
    if data.get("method") != "manual_clicker":
        logger.warning("Found phantom_calibration.json but not from manual_clicker")
        return None

    # Extract calibration parameters
    cal = data.get("calibration", {})
    rods_data = data.get("rods", [])
    qc_data = data.get("qc", {})

    # Convert rod data to PhantomRod objects
    rods = []
    for rod in rods_data:
        center = rod.get("center_voxel", [0, 0, 0])
        phantom_rod = PhantomRod(
            name=rod["name"],
            known_density_mgcm3=rod["density_mgcm3"],
            center_voxel=tuple(center),
            radius_voxels=4.0,  # Default from manual calibration
            measured_hu_mean=rod["mean_hu"],
            measured_hu_std=rod.get("std_hu", 0.0),
            voxel_count=rod.get("voxel_count", 0)
        )
        rods.append(phantom_rod)

    return CalibrationResult(
        success=True,
        slope=cal.get("slope", 0.0),
        intercept=cal.get("intercept", 0.0),
        r_squared=cal.get("r_squared", 0.0),
        # drift_correction_hu is -drift_hu, i.e., what to ADD to correct scanner drift
        drift_offset_hu=cal.get("drift_correction_hu", 0.0),
        rods=rods,
        qc_passed=qc_data.get("passed", False),
        qc_messages=qc_data.get("messages", ["Loaded from manual calibration"])
    )


def calibrate_phantom(ct_path: Path, output_dir: Path,
                      manual_override: Optional[dict] = None,
                      prefer_manual: bool = True) -> CalibrationResult:
    """
    Run complete phantom calibration for a CT scan.

    First checks for manual calibration (from manual_calibration.py).
    If not found, attempts automated detection.

    Args:
        ct_path: Path to CT NIfTI file
        output_dir: Directory for calibration outputs
        manual_override: Optional dict with 'rod_centers_voxel' list of (x, y) centers
        prefer_manual: If True, use manual calibration if available (default: True)

    Returns:
        CalibrationResult dataclass with calibration parameters
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for manual calibration first
    if prefer_manual:
        manual_result = load_manual_calibration(output_dir)
        if manual_result is not None:
            logger.info("Using manual calibration (from manual_calibration.py)")
            return manual_result

    logger.info("No manual calibration found, attempting automated detection...")

    # Load CT
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()
    voxel_sizes = ct_nii.header.get_zooms()

    logger.info(f"CT loaded: shape={ct_data.shape}, voxel_sizes={voxel_sizes}")

    # Find phantom slices
    z_start, z_end = find_phantom_slices_simple(ct_data)
    logger.info(f"Phantom z-range: {z_start} to {z_end}")

    # Use middle slice for circle detection
    z_mid = (z_start + z_end) // 2
    phantom_slice = ct_data[:, :, z_mid]

    # Detect circles
    if manual_override and "rod_centers_voxel" in manual_override:
        logger.info("Using manual phantom rod positions")
        circles = [(c[0], c[1], 10) for c in manual_override["rod_centers_voxel"]]
    else:
        circles = detect_phantom_circles(phantom_slice,
                                         voxel_size_xy=float(voxel_sizes[0]))

    logger.info(f"Detected {len(circles)} circles in phantom slice")

    if len(circles) < 4:
        logger.error("Insufficient circles detected for calibration")
        return CalibrationResult(
            success=False,
            slope=0.0,
            intercept=0.0,
            r_squared=0.0,
            drift_offset_hu=0.0,
            rods=[],
            qc_passed=False,
            qc_messages=["FAIL: Insufficient phantom circles detected (<4)"]
        )

    # Identify rods
    rods = identify_rods_by_hu(circles, ct_data, (z_start, z_end))

    if len(rods) < 4:
        logger.error(f"Could not identify sufficient rods: {list(rods.keys())}")
        return CalibrationResult(
            success=False,
            slope=0.0,
            intercept=0.0,
            r_squared=0.0,
            drift_offset_hu=0.0,
            rods=list(rods.values()),
            qc_passed=False,
            qc_messages=[f"FAIL: Only identified {len(rods)} rods"]
        )

    # Log identified rods
    for name, rod in rods.items():
        logger.info(f"  {name}: HU={rod.measured_hu_mean:.1f}±{rod.measured_hu_std:.1f}, "
                    f"density={rod.known_density_mgcm3} mg/cm³")

    # Compute calibration
    slope, intercept, r_squared = compute_calibration(rods)

    # Get drift offset from base rod
    # Base rod (water-equivalent, 0 mg/cm³) should read ~0 HU
    # If it reads +10 HU, scanner is reading 10 HU too high
    # Drift correction = what to ADD to correct: if base reads +10, add -10
    base_drift_hu = rods["base"].measured_hu_mean if "base" in rods else 0.0
    drift_offset = -base_drift_hu  # What to ADD to correct scanner drift

    # Run QC
    qc_passed, qc_messages = run_qc_checks(rods, slope, intercept, r_squared)

    # Save calibration files
    bmd_cal = {
        "purpose": "HU to mg/cm³ for bone mineral density",
        "calibration_points": {},
        "regression": {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared
        },
        "phantom_z_range": [z_start, z_end]
    }

    for name, rod in rods.items():
        if name != "fat":
            bmd_cal["calibration_points"][name] = {
                "hu": rod.measured_hu_mean,
                "bmd_mgcm3": rod.known_density_mgcm3
            }

    with open(output_dir / "calibration_bmd.json", "w") as f:
        json.dump(bmd_cal, f, indent=2)

    # HU stability calibration for muscle
    hu_stability = {
        "purpose": "Scanner drift correction for HU-based classification",
        "drift_correction": {"offset_hu": drift_offset},
        "scale_stability": {
            "rod_spread_hu": (rods["bone_200"].measured_hu_mean - rods["base"].measured_hu_mean)
            if "bone_200" in rods else 0.0
        }
    }

    with open(output_dir / "calibration_hu_stability.json", "w") as f:
        json.dump(hu_stability, f, indent=2)

    # Save phantom ROI masks for QC
    phantom_masks_dir = output_dir / "phantom_rois"
    phantom_masks_dir.mkdir(parents=True, exist_ok=True)

    roi_mask = np.zeros(ct_data.shape, dtype=np.uint8)

    for i, (name, rod) in enumerate(rods.items(), start=1):
        cx, cy, _ = rod.center_voxel
        radius = int(rod.radius_voxels)

        for z in range(z_start, z_end + 1):
            y_grid, x_grid = np.ogrid[:ct_data.shape[0], :ct_data.shape[1]]
            dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            roi_mask[:, :, z][dist_from_center <= radius] = i

    roi_nii = nib.Nifti1Image(roi_mask, ct_nii.affine, ct_nii.header)
    nib.save(roi_nii, phantom_masks_dir / "phantom_rois_combined.nii.gz")

    for i, (name, rod) in enumerate(rods.items(), start=1):
        individual_mask = (roi_mask == i).astype(np.uint8)
        individual_nii = nib.Nifti1Image(individual_mask, ct_nii.affine, ct_nii.header)
        nib.save(individual_nii, phantom_masks_dir / f"phantom_roi_{name}.nii.gz")

    rod_info = {
        "z_range": [z_start, z_end],
        "rods": {}
    }
    for name, rod in rods.items():
        rod_info["rods"][name] = {
            "center_voxel": list(rod.center_voxel),
            "radius_voxels": rod.radius_voxels,
            "measured_hu_mean": rod.measured_hu_mean,
            "measured_hu_std": rod.measured_hu_std,
            "known_density_mgcm3": rod.known_density_mgcm3,
            "voxel_count": rod.voxel_count
        }

    with open(phantom_masks_dir / "phantom_rod_info.json", "w") as f:
        json.dump(rod_info, f, indent=2)

    logger.info(f"Phantom ROI masks saved to {phantom_masks_dir}")
    logger.info(f"Calibration saved to {output_dir}")

    return CalibrationResult(
        success=True,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        drift_offset_hu=drift_offset,
        rods=list(rods.values()),
        qc_passed=qc_passed,
        qc_messages=qc_messages
    )


def load_calibration(calibration_dir: Path) -> Tuple[float, float, float]:
    """
    Load previously computed calibration.
    """
    bmd_path = calibration_dir / "calibration_bmd.json"
    stability_path = calibration_dir / "calibration_hu_stability.json"

    with open(bmd_path) as f:
        bmd_cal = json.load(f)

    with open(stability_path) as f:
        stability_cal = json.load(f)

    slope = bmd_cal["regression"]["slope"]
    intercept = bmd_cal["regression"]["intercept"]
    drift_offset = stability_cal["drift_correction"]["offset_hu"]

    return slope, intercept, drift_offset


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python 02_phantom_calibration.py <ct_path> <output_dir>")
        sys.exit(1)

    ct_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    result = calibrate_phantom(ct_path, output_dir)

    print(f"\nCalibration {'successful' if result.success else 'failed'}")
    print(f"Slope: {result.slope:.4f}")
    print(f"Intercept: {result.intercept:.2f}")
    print(f"R²: {result.r_squared:.4f}")
    print(f"Drift: {result.drift_offset_hu:.1f} HU")
    print(f"QC: {'PASSED' if result.qc_passed else 'FAILED'}")
    for msg in result.qc_messages:
        print(f"  {msg}")
