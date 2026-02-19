#!/usr/bin/env python3
"""
manual_calibration.py - Semi-automated manual phantom calibration tool

This script provides an interactive interface for manually clicking on phantom
rod centers when automated detection fails. It samples cylindrical ROIs around
each click and computes the HU→BMD calibration.

Usage:
    python manual_calibration.py --data ../RawData/bmd_ct --output ..
    python manual_calibration.py --data ../RawData/bmd_ct --output .. --force
    python manual_calibration.py --data ../RawData/bmd_ct --output .. --subject sub-101

Rod clicking order:
    1. Fat (-100 mg/cm³ equivalent) - for QC only, excluded from regression
    2. Base/Water (0 mg/cm³) - the plastic base material
    3. Low Bone (50 mg/cm³)
    4. Mid Bone (100 mg/cm³)
    5. High Bone (200 mg/cm³)
"""

import matplotlib
matplotlib.use('TkAgg')  # Interactive backend

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
from scipy.stats import linregress


# ============================================================================
# CONFIGURATION
# ============================================================================

# Rod definitions - order of clicking
# Note: Fat rod is -100 mg/cm³ equivalent on Mindways phantom (not -50)
ROD_DEFINITIONS = [
    {"name": "fat", "label": "Fat (-100)", "density_mgcm3": -100, "use_in_regression": False},
    {"name": "base", "label": "Base (0)", "density_mgcm3": 0, "use_in_regression": True},
    {"name": "bone_50", "label": "Low Bone (50)", "density_mgcm3": 50, "use_in_regression": True},
    {"name": "bone_100", "label": "Mid Bone (100)", "density_mgcm3": 100, "use_in_regression": True},
    {"name": "bone_200", "label": "High Bone (200)", "density_mgcm3": 200, "use_in_regression": True},
]

# Sampling parameters
SAMPLE_RADIUS_MM = 4.0      # Radius of cylindrical ROI in mm
SAMPLE_Z_SLICES = 4         # Number of slices above and below (total = 2*N + 1 = 9)

# QC thresholds
MIN_R_SQUARED = 0.95        # Minimum acceptable R² for calibration
MAX_DRIFT_HU = 50           # Maximum acceptable drift in HU

# Phantom location (typically in inferior slices for supine spine CT)
PHANTOM_Z_SEARCH_START = 0
PHANTOM_Z_SEARCH_END = 80   # Slices to search for phantom
DEFAULT_Z_SLICE = 50        # Default starting Z slice for display


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def find_ct_scans(data_dir: Path, subject_filter: Optional[str] = None) -> List[Tuple[str, str, Path]]:
    """
    Find all CT scans to process.

    Returns:
        List of (subject_id, session, ct_path) tuples
    """
    scans = []

    subject_dirs = sorted([d for d in data_dir.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        if subject_filter and subject_id != subject_filter:
            continue

        for session in ["ses-Baseline", "ses-Followup"]:
            session_dir = subject_dir / session / "ct"

            if not session_dir.exists():
                continue

            # Find 1.25mm CT (preferred resolution)
            ct_files = list(session_dir.glob("*_rec-stnd1.25mm_ct.nii.gz"))

            if ct_files:
                scans.append((subject_id, session, ct_files[0]))

    return scans


def get_core_sample(ct_data: np.ndarray,
                    center_i: int,
                    center_j: int,
                    center_z: int,
                    voxel_sizes: Tuple[float, float, float],
                    radius_mm: float = SAMPLE_RADIUS_MM,
                    z_slices: int = SAMPLE_Z_SLICES) -> Tuple[float, float, int, np.ndarray]:
    """
    Sample a cylindrical ROI around the clicked point.

    Coordinate convention:
    - center_i: Index along ct_data axis 0
    - center_j: Index along ct_data axis 1
    - center_z: Index along ct_data axis 2 (z-slice)
    - The mask will be True at ct_data[i, j, z] for voxels within the cylinder

    Args:
        ct_data: 3D CT volume (shape: dim0 x dim1 x dim2)
        center_i: Index along axis 0
        center_j: Index along axis 1
        center_z: Index along axis 2 (z-slice)
        voxel_sizes: Voxel dimensions in mm
        radius_mm: Radius of cylinder in mm
        z_slices: Number of slices above/below center

    Returns:
        Tuple of (mean_hu, std_hu, voxel_count, 3d_mask)
    """
    # Convert radius from mm to voxels (use average of x,y for in-plane)
    radius_voxels = radius_mm / np.mean(voxel_sizes[:2])

    # Z range
    z_min = max(0, center_z - z_slices)
    z_max = min(ct_data.shape[2], center_z + z_slices + 1)

    # Create cylindrical mask
    mask = np.zeros(ct_data.shape, dtype=bool)

    # Create 2D circular mask template
    # ii indexes axis 0 (shape[0]), jj indexes axis 1 (shape[1])
    ii, jj = np.ogrid[:ct_data.shape[0], :ct_data.shape[1]]
    circle_mask = ((ii - center_i)**2 + (jj - center_j)**2) <= radius_voxels**2

    # Apply to each z-slice
    for z in range(z_min, z_max):
        mask[:, :, z] = circle_mask

    # Extract values
    values = ct_data[mask]

    if len(values) == 0:
        return 0.0, 0.0, 0, mask

    return float(np.mean(values)), float(np.std(values)), len(values), mask


def compute_calibration(rod_results: List[Dict]) -> Dict:
    """
    Compute linear regression for HU → BMD calibration.

    Uses only rods marked for regression (excludes Fat).

    Args:
        rod_results: List of rod measurement dictionaries

    Returns:
        Dictionary with calibration parameters
    """
    # Extract regression points (exclude fat)
    hu_values = []
    bmd_values = []

    for rod in rod_results:
        if rod.get("use_in_regression", True):
            hu_values.append(rod["mean_hu"])
            bmd_values.append(rod["density_mgcm3"])

    if len(hu_values) < 3:
        return {
            "success": False,
            "error": "Insufficient points for regression"
        }

    # Linear regression: BMD = slope * HU + intercept
    slope, intercept, r_value, p_value, std_err = linregress(hu_values, bmd_values)
    r_squared = r_value ** 2

    # Get base (water) HU for drift calculation
    base_rod = next((r for r in rod_results if r["name"] == "base"), None)
    drift_hu = base_rod["mean_hu"] if base_rod else 0.0

    # Drift correction: what to ADD to measured HU to correct
    # If base reads -10 HU, we need to add +10 to correct
    drift_correction_hu = -drift_hu

    return {
        "success": True,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "std_err": std_err,
        "drift_hu": drift_hu,
        "drift_correction_hu": drift_correction_hu
    }


def run_qc_checks(calibration: Dict, rod_results: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Run quality control checks on calibration.

    Returns:
        Tuple of (passed, list_of_messages)
    """
    messages = []
    passed = True

    # Check R²
    r_squared = calibration.get("r_squared", 0)
    if r_squared < MIN_R_SQUARED:
        messages.append(f"WARNING: R² = {r_squared:.4f} < {MIN_R_SQUARED}")
        if r_squared < 0.90:
            passed = False
            messages.append("FAIL: R² too low for reliable calibration")
    else:
        messages.append(f"OK: R² = {r_squared:.4f}")

    # Check drift
    drift_hu = abs(calibration.get("drift_hu", 0))
    if drift_hu > MAX_DRIFT_HU:
        messages.append(f"FAIL: Drift = {drift_hu:.1f} HU > {MAX_DRIFT_HU} HU threshold")
        passed = False
    elif drift_hu > 30:
        messages.append(f"WARNING: Drift = {drift_hu:.1f} HU > 30 HU")
    else:
        messages.append(f"OK: Drift = {calibration.get('drift_hu', 0):.1f} HU")

    # Check monotonic ordering of HU values
    regression_rods = [r for r in rod_results if r.get("use_in_regression", True)]
    hu_ordered = [r["mean_hu"] for r in sorted(regression_rods, key=lambda x: x["density_mgcm3"])]

    if hu_ordered != sorted(hu_ordered):
        messages.append("FAIL: HU values are not monotonically increasing with density")
        passed = False
    else:
        messages.append("OK: HU values monotonically increasing")

    # Check individual rod residuals
    slope = calibration.get("slope", 1)
    intercept = calibration.get("intercept", 0)

    for rod in regression_rods:
        predicted = slope * rod["mean_hu"] + intercept
        residual = abs(predicted - rod["density_mgcm3"])
        residual_hu = residual / slope if slope != 0 else float('inf')

        if residual_hu > 15:
            messages.append(f"WARNING: {rod['name']} residual = {residual_hu:.1f} HU")
        else:
            messages.append(f"OK: {rod['name']} residual = {residual_hu:.1f} HU")

    return passed, messages


# ============================================================================
# INTERACTIVE GUI
# ============================================================================

class PhantomClicker:
    """Interactive matplotlib-based phantom rod clicker."""

    def __init__(self, ct_data: np.ndarray, voxel_sizes: Tuple[float, float, float],
                 subject_id: str, session: str):
        self.ct_data = ct_data
        self.voxel_sizes = voxel_sizes
        self.subject_id = subject_id
        self.session = session

        # State
        self.current_z = self._find_phantom_slice()
        self.clicks = []
        self.current_rod_idx = 0
        self.completed = False
        self.aborted = False
        self.quit_all = False  # Flag to quit entire application

        # Visual elements
        self.circles = []
        self.texts = []

    def _find_phantom_slice(self) -> int:
        """Find a good starting slice in the phantom region."""
        # Default to Z=50 as specified
        default_z = min(DEFAULT_Z_SLICE, self.ct_data.shape[2] - 1)

        # Verify it has content, otherwise search
        if np.percentile(self.ct_data[:, :, default_z], 75) > -500:
            return default_z

        # Fallback: search for slice with content
        z_end = min(PHANTOM_Z_SEARCH_END, self.ct_data.shape[2])
        for z in range(PHANTOM_Z_SEARCH_START, z_end):
            slice_data = self.ct_data[:, :, z]
            if np.percentile(slice_data, 75) > -500:
                return z

        return default_z

    def run(self) -> Tuple[Optional[List[Tuple[int, int, int]]], bool]:
        """
        Run the interactive clicker.

        Returns:
            Tuple of (clicks, quit_all):
            - clicks: List of (x, y, z) coordinates for each rod, or None if aborted
            - quit_all: True if user requested to quit entire application
        """
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2)

        # Display initial slice
        # NOTE: We do NOT transpose here. This means:
        # - imshow displays array with rows=axis0, cols=axis1
        # - Click at (xdata, ydata) corresponds to ct_data[ydata, xdata, z]
        # - This is because imshow puts axis 0 as vertical (y) and axis 1 as horizontal (x)
        self.im = self.ax.imshow(
            self.ct_data[:, :, self.current_z],
            cmap='gray',
            vmin=-200,
            vmax=400,
            origin='lower',
            aspect='equal'
        )

        self._update_title()

        # Add slider for z navigation
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.z_slider = Slider(
            ax_slider, 'Z Slice',
            0, self.ct_data.shape[2] - 1,
            valinit=self.current_z,
            valstep=1
        )
        self.z_slider.on_changed(self._on_slider_change)

        # Add buttons
        ax_undo = plt.axes([0.15, 0.02, 0.12, 0.05])
        ax_done = plt.axes([0.30, 0.02, 0.12, 0.05])
        ax_abort = plt.axes([0.45, 0.02, 0.12, 0.05])
        ax_quit = plt.axes([0.60, 0.02, 0.15, 0.05])

        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_done = Button(ax_done, 'Done')
        self.btn_abort = Button(ax_abort, 'Skip')
        self.btn_quit = Button(ax_quit, 'QUIT ALL')

        # Style the quit button to stand out
        self.btn_quit.color = 'lightcoral'
        self.btn_quit.hovercolor = 'red'

        self.btn_undo.on_clicked(self._on_undo)
        self.btn_done.on_clicked(self._on_done)
        self.btn_abort.on_clicked(self._on_abort)
        self.btn_quit.on_clicked(self._on_quit_all)

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Instructions
        print(f"\n{'='*60}")
        print(f"Manual Phantom Calibration: {self.subject_id} / {self.session}")
        print(f"{'='*60}")
        print("Instructions:")
        print("  - Use slider or up/down arrows to navigate Z slices")
        print("  - LEFT CLICK on each rod center in order")
        print("  - Press 'u' or click 'Undo' to undo last click")
        print("  - Press 'Enter' or click 'Done' when finished")
        print("  - Press 'Escape' or click 'Skip' to skip this scan")
        print("  - Click 'QUIT ALL' or press 'q' to exit entirely")
        print(f"{'='*60}\n")

        plt.show()

        if self.aborted or not self.completed:
            return None, self.quit_all

        return self.clicks, self.quit_all

    def _update_title(self):
        """Update figure title with current state."""
        if self.current_rod_idx < len(ROD_DEFINITIONS):
            rod = ROD_DEFINITIONS[self.current_rod_idx]
            instruction = f"Click: {rod['label']}"
        else:
            instruction = "All rods clicked! Press Enter or click Done."

        self.ax.set_title(
            f"{self.subject_id} / {self.session}\n"
            f"Z = {self.current_z} | {instruction}\n"
            f"Clicked: {self.current_rod_idx}/{len(ROD_DEFINITIONS)} rods"
        )
        self.fig.canvas.draw_idle()

    def _update_display(self):
        """Update the displayed slice."""
        self.im.set_data(self.ct_data[:, :, self.current_z])
        self._update_title()
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, val):
        """Handle slider change."""
        self.current_z = int(val)
        self._update_display()

    def _on_click(self, event):
        """Handle mouse click."""
        if event.inaxes != self.ax:
            return

        if event.button != 1:  # Left click only
            return

        if self.current_rod_idx >= len(ROD_DEFINITIONS):
            return

        # Get click coordinates from matplotlib event
        # We display ct_data[:,:,z] WITHOUT transpose, with origin='lower'
        #
        # Coordinate system (no transpose):
        # - ct_data has shape (dim0, dim1, dim2)
        # - ct_data[:,:,z] has shape (dim0, dim1)
        # - imshow displays 2D array with:
        #     - rows (vertical, y-axis) = axis 0 of array
        #     - cols (horizontal, x-axis) = axis 1 of array
        # - With origin='lower', row 0 (array index 0) is at bottom (y=0)
        #
        # matplotlib event coordinates:
        # - event.xdata = horizontal position = column = axis 1 of array
        # - event.ydata = vertical position = row = axis 0 of array
        #
        # Therefore: ct_data[int(ydata), int(xdata), z] is the clicked voxel
        # - data_i (axis 0) = ydata (vertical/row)
        # - data_j (axis 1) = xdata (horizontal/col)

        display_x = int(event.xdata)  # horizontal position in display = col = axis 1
        display_y = int(event.ydata)  # vertical position in display = row = axis 0
        z = self.current_z

        # Convert display coords to data array indices
        data_i = display_y  # vertical (row) → axis 0 of ct_data
        data_j = display_x  # horizontal (col) → axis 1 of ct_data

        # Store click - these are DATA indices: ct_data[data_i, data_j, z]
        self.clicks.append((data_i, data_j, z))

        # For visual feedback on the DISPLAY, use display coordinates
        x, y = display_x, display_y

        # Visual feedback
        rod = ROD_DEFINITIONS[self.current_rod_idx]
        radius_px = SAMPLE_RADIUS_MM / np.mean(self.voxel_sizes[:2])

        circle = Circle(
            (x, y), radius=radius_px,
            color='red', fill=False, linewidth=2
        )
        self.ax.add_patch(circle)
        self.circles.append(circle)

        text = self.ax.text(
            x + radius_px + 2, y,
            rod['label'],
            color='yellow',
            fontsize=10,
            fontweight='bold'
        )
        self.texts.append(text)

        print(f"  Clicked {rod['label']}: ({x}, {y}, {z})")

        self.current_rod_idx += 1
        self._update_title()

    def _on_key(self, event):
        """Handle key press."""
        if event.key == 'u':
            self._on_undo(None)
        elif event.key == 'enter':
            self._on_done(None)
        elif event.key == 'escape':
            self._on_abort(None)
        elif event.key == 'q':
            self._on_quit_all(None)
        elif event.key == 'up':
            self.current_z = min(self.current_z + 1, self.ct_data.shape[2] - 1)
            self.z_slider.set_val(self.current_z)
        elif event.key == 'down':
            self.current_z = max(self.current_z - 1, 0)
            self.z_slider.set_val(self.current_z)

    def _on_undo(self, event):
        """Undo last click."""
        if self.clicks:
            self.clicks.pop()

            if self.circles:
                self.circles[-1].remove()
                self.circles.pop()

            if self.texts:
                self.texts[-1].remove()
                self.texts.pop()

            self.current_rod_idx = max(0, self.current_rod_idx - 1)
            self._update_title()
            print("  Undo: removed last click")

    def _on_done(self, event):
        """Mark as done."""
        if self.current_rod_idx < len(ROD_DEFINITIONS):
            print(f"  Warning: Only {self.current_rod_idx}/{len(ROD_DEFINITIONS)} rods clicked!")
            return

        self.completed = True
        plt.close(self.fig)

    def _on_abort(self, event):
        """Skip this scan (abort calibration for current subject)."""
        self.aborted = True
        plt.close(self.fig)
        print("  Skipped by user")

    def _on_quit_all(self, event):
        """Quit the entire application."""
        self.quit_all = True
        self.aborted = True
        plt.close(self.fig)
        print("  QUIT ALL requested - exiting application")


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def save_calibration_outputs(output_dir: Path,
                             qc_dir: Path,
                             ct_nii: nib.Nifti1Image,
                             clicks: List[Tuple[int, int, int]],
                             rod_results: List[Dict],
                             calibration: Dict,
                             qc_passed: bool,
                             qc_messages: List[str]):
    """
    Save all calibration outputs.

    Generates in output_dir (DerivedData):
    - phantom_calibration.json (main calibration file)
    - calibration_bmd.json (pipeline-compatible BMD calibration)
    - calibration_hu_stability.json (pipeline-compatible drift correction)
    - phantom_rois/phantom_rois_combined.nii.gz (labeled mask of ROIs)
    - phantom_rois/phantom_roi_*.nii.gz (individual rod masks)

    Generates in qc_dir (QC):
    - phantom_clicks.png (QC image of clicked positions)
    - calibration_curve.png (QC regression plot)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    ct_data = ct_nii.get_fdata()
    voxel_sizes = ct_nii.header.get_zooms()[:3]

    # 1. Main calibration JSON
    main_cal = {
        "method": "manual_clicker",
        "timestamp": datetime.now().isoformat(),
        "calibration": {
            "slope": calibration["slope"],
            "intercept": calibration["intercept"],
            "r_squared": calibration["r_squared"],
            "drift_hu": calibration["drift_hu"],
            "drift_correction_hu": calibration["drift_correction_hu"]
        },
        "rods": rod_results,
        "qc": {
            "passed": qc_passed,
            "messages": qc_messages
        },
        "parameters": {
            "sample_radius_mm": SAMPLE_RADIUS_MM,
            "sample_z_slices": SAMPLE_Z_SLICES
        }
    }

    with open(output_dir / "phantom_calibration.json", "w") as f:
        json.dump(main_cal, f, indent=2)

    # 2. Pipeline-compatible calibration_bmd.json
    bmd_cal = {
        "purpose": "HU to mg/cm³ for bone mineral density",
        "method": "manual_clicker",
        "calibration_points": {},
        "regression": {
            "slope": calibration["slope"],
            "intercept": calibration["intercept"],
            "r_squared": calibration["r_squared"]
        },
        "phantom_z_range": [
            min(c[2] for c in clicks) - SAMPLE_Z_SLICES,
            max(c[2] for c in clicks) + SAMPLE_Z_SLICES
        ]
    }

    for rod in rod_results:
        if rod["name"] != "fat":
            bmd_cal["calibration_points"][rod["name"]] = {
                "hu": rod["mean_hu"],
                "bmd_mgcm3": rod["density_mgcm3"]
            }

    with open(output_dir / "calibration_bmd.json", "w") as f:
        json.dump(bmd_cal, f, indent=2)

    # 3. Pipeline-compatible calibration_hu_stability.json
    hu_stability = {
        "purpose": "Scanner drift correction for HU-based classification",
        "method": "manual_clicker",
        "drift_correction": {
            "offset_hu": calibration["drift_correction_hu"]
        },
        "scale_stability": {
            "rod_spread_hu": rod_results[-1]["mean_hu"] - rod_results[1]["mean_hu"]
            if len(rod_results) >= 5 else 0.0
        }
    }

    with open(output_dir / "calibration_hu_stability.json", "w") as f:
        json.dump(hu_stability, f, indent=2)

    # 4. Create and save rod masks
    combined_mask = np.zeros(ct_data.shape, dtype=np.uint8)

    for i, (rod, (cx, cy, cz)) in enumerate(zip(rod_results, clicks)):
        _, _, _, mask = get_core_sample(ct_data, cx, cy, cz, voxel_sizes)
        combined_mask[mask] = i + 1  # Label 1-5

    # Save phantom ROI directory
    phantom_rois_dir = output_dir / "phantom_rois"
    phantom_rois_dir.mkdir(parents=True, exist_ok=True)

    mask_nii = nib.Nifti1Image(combined_mask, ct_nii.affine, ct_nii.header)
    nib.save(mask_nii, phantom_rois_dir / "phantom_rois_combined.nii.gz")

    # Save individual rod masks
    for i, rod in enumerate(rod_results):
        individual_mask = (combined_mask == (i + 1)).astype(np.uint8)
        individual_nii = nib.Nifti1Image(individual_mask, ct_nii.affine, ct_nii.header)
        nib.save(individual_nii, phantom_rois_dir / f"phantom_roi_{rod['name']}.nii.gz")

    # Save rod info JSON
    rod_info = {
        "z_range": [
            min(c[2] for c in clicks) - SAMPLE_Z_SLICES,
            max(c[2] for c in clicks) + SAMPLE_Z_SLICES
        ],
        "rods": {}
    }
    for rod, (cx, cy, cz) in zip(rod_results, clicks):
        rod_info["rods"][rod["name"]] = {
            "center_voxel": [cx, cy, cz],
            "radius_mm": SAMPLE_RADIUS_MM,
            "measured_hu_mean": rod["mean_hu"],
            "measured_hu_std": rod["std_hu"],
            "known_density_mgcm3": rod["density_mgcm3"],
            "voxel_count": rod["voxel_count"]
        }

    with open(phantom_rois_dir / "phantom_rod_info.json", "w") as f:
        json.dump(rod_info, f, indent=2)

    # 5. QC Images (saved to QC folder)
    _save_phantom_clicks_image(ct_data, clicks, rod_results, voxel_sizes,
                               qc_dir / "phantom_clicks.png")

    _save_calibration_curve_image(rod_results, calibration,
                                   qc_dir / "calibration_curve.png")

    print(f"  Saved calibration to: {output_dir}")
    print(f"  Saved QC images to: {qc_dir}")


def _save_phantom_clicks_image(ct_data: np.ndarray,
                                clicks: List[Tuple[int, int, int]],
                                rod_results: List[Dict],
                                voxel_sizes: Tuple[float, float, float],
                                output_path: Path):
    """Save QC image showing clicked positions."""
    # Use the z-slice of the first click
    z_mid = clicks[0][2] if clicks else ct_data.shape[2] // 2

    fig, ax = plt.subplots(figsize=(10, 10))

    # Display CT slice (same as interactive display - NO transpose)
    ax.imshow(ct_data[:, :, z_mid], cmap='gray', vmin=-200, vmax=400, origin='lower')

    # Draw circles at click positions
    # clicks are stored as (data_i, data_j, z) where ct_data[data_i, data_j, z]
    # With no transpose: data_i = display_y (row), data_j = display_x (col)
    # So: display_x = data_j, display_y = data_i
    radius_px = SAMPLE_RADIUS_MM / np.mean(voxel_sizes[:2])
    colors = ['purple', 'blue', 'green', 'orange', 'red']

    for i, ((data_i, data_j, cz), rod) in enumerate(zip(clicks, rod_results)):
        color = colors[i % len(colors)]

        # Convert data coords back to display coords (no transpose)
        # data_i = row = y, data_j = col = x
        display_x = data_j  # horizontal position (column)
        display_y = data_i  # vertical position (row)

        # Circle at display coordinates
        circle = Circle((display_x, display_y), radius=radius_px, color=color, fill=False, linewidth=2)
        ax.add_patch(circle)

        # Label
        ax.text(display_x + radius_px + 3, display_y,
                f"{rod['label']}\nHU={rod['mean_hu']:.1f}",
                color=color, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title(f"Manual Phantom Calibration (z={z_mid})\n"
                 f"Click positions and measured HU values")
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Y (voxels)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _save_calibration_curve_image(rod_results: List[Dict],
                                   calibration: Dict,
                                   output_path: Path):
    """Save QC image of calibration regression."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all points
    for rod in rod_results:
        color = 'gray' if not rod.get("use_in_regression", True) else 'blue'
        marker = 'x' if not rod.get("use_in_regression", True) else 'o'
        ax.scatter(rod["mean_hu"], rod["density_mgcm3"],
                   s=100, c=color, marker=marker, zorder=5)
        ax.annotate(rod["name"], (rod["mean_hu"], rod["density_mgcm3"]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Plot regression line
    slope = calibration["slope"]
    intercept = calibration["intercept"]
    r_squared = calibration["r_squared"]

    regression_rods = [r for r in rod_results if r.get("use_in_regression", True)]
    hu_min = min(r["mean_hu"] for r in regression_rods) - 20
    hu_max = max(r["mean_hu"] for r in regression_rods) + 20

    x_line = np.linspace(hu_min, hu_max, 100)
    y_line = slope * x_line + intercept

    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f'BMD = {slope:.4f} × HU + {intercept:.2f}')

    ax.set_xlabel("HU (Hounsfield Units)")
    ax.set_ylabel("BMD (mg/cm³)")
    ax.set_title(f"Phantom Calibration Curve\n"
                 f"R² = {r_squared:.4f} | Drift = {calibration['drift_hu']:.1f} HU")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def process_scan(subject_id: str, session: str, ct_path: Path,
                 output_base: Path, force: bool = False) -> Tuple[bool, bool]:
    """
    Process a single CT scan with manual calibration.

    Returns:
        Tuple of (success, quit_all):
        - success: True if calibration completed successfully
        - quit_all: True if user requested to quit entire application
    """
    # Set up output directory
    derived_dir = output_base / "DerivedData" / subject_id / session

    # Check if already calibrated
    existing_cal = derived_dir / "phantom_calibration.json"
    if existing_cal.exists() and not force:
        print(f"  Skipping: calibration already exists (use --force to overwrite)")
        return True, False

    # Load CT
    print(f"\n  Loading CT: {ct_path}")
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata()
    voxel_sizes = ct_nii.header.get_zooms()[:3]

    print(f"  Shape: {ct_data.shape}, Voxel size: {voxel_sizes}")

    # Run interactive clicker
    clicker = PhantomClicker(ct_data, voxel_sizes, subject_id, session)
    clicks, quit_all = clicker.run()

    if quit_all:
        return False, True

    if clicks is None:
        print("  Calibration skipped")
        return False, False

    if len(clicks) != len(ROD_DEFINITIONS):
        print(f"  Error: Expected {len(ROD_DEFINITIONS)} clicks, got {len(clicks)}")
        return False, False

    # Sample each rod
    print("\n  Sampling ROIs...")
    rod_results = []

    for i, (click, rod_def) in enumerate(zip(clicks, ROD_DEFINITIONS)):
        cx, cy, cz = click
        mean_hu, std_hu, count, _ = get_core_sample(ct_data, cx, cy, cz, voxel_sizes)

        rod_result = {
            "name": rod_def["name"],
            "label": rod_def["label"],
            "density_mgcm3": rod_def["density_mgcm3"],
            "use_in_regression": rod_def["use_in_regression"],
            "mean_hu": mean_hu,
            "std_hu": std_hu,
            "voxel_count": count,
            "center_voxel": [cx, cy, cz]
        }
        rod_results.append(rod_result)

        print(f"    {rod_def['label']}: {mean_hu:.1f} ± {std_hu:.1f} HU (n={count})")

    # Compute calibration
    print("\n  Computing calibration...")
    calibration = compute_calibration(rod_results)

    if not calibration["success"]:
        print(f"  Error: {calibration.get('error', 'Unknown')}")
        return False, False

    print(f"    Slope: {calibration['slope']:.4f}")
    print(f"    Intercept: {calibration['intercept']:.2f}")
    print(f"    R²: {calibration['r_squared']:.4f}")
    print(f"    Drift: {calibration['drift_hu']:.1f} HU (correction: {calibration['drift_correction_hu']:.1f})")

    # QC checks
    print("\n  QC Checks:")
    qc_passed, qc_messages = run_qc_checks(calibration, rod_results)
    for msg in qc_messages:
        print(f"    {msg}")

    print(f"\n  QC Status: {'PASSED' if qc_passed else 'FAILED'}")

    # Set up QC directory
    qc_dir = output_base / "QC" / subject_id / session
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    print("\n  Saving outputs...")
    save_calibration_outputs(derived_dir, qc_dir, ct_nii, clicks, rod_results,
                             calibration, qc_passed, qc_messages)

    return True, False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manual phantom calibration tool for ERAP CT analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Rod clicking order:
  1. Fat (-100 mg/cm³) - purple, QC only
  2. Base (0 mg/cm³) - blue, water reference
  3. Low Bone (50 mg/cm³) - green
  4. Mid Bone (100 mg/cm³) - orange
  5. High Bone (200 mg/cm³) - red

Examples:
  Process all subjects:
    python manual_calibration.py --data ../RawData/bmd_ct --output ..

  Process single subject:
    python manual_calibration.py --data ../RawData/bmd_ct --output .. --subject sub-101

  Force re-calibration:
    python manual_calibration.py --data ../RawData/bmd_ct --output .. --force
        """
    )

    parser.add_argument("--data", required=True, type=Path,
                        help="Path to raw data directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to output directory")
    parser.add_argument("--subject", type=str, default=None,
                        help="Process only this subject")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing calibrations")

    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: Data directory not found: {args.data}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # Find scans
    scans = find_ct_scans(args.data, args.subject)

    if not scans:
        print("No CT scans found to process")
        sys.exit(1)

    print(f"\nFound {len(scans)} CT scans to process:")
    for subject_id, session, ct_path in scans:
        print(f"  {subject_id} / {session}")

    # Process each scan
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, (subject_id, session, ct_path) in enumerate(scans, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(scans)}] {subject_id} / {session}")
        print(f"{'='*60}")

        # Check if already exists
        derived_dir = args.output / "DerivedData" / subject_id / session
        existing_cal = derived_dir / "phantom_calibration.json"

        if existing_cal.exists() and not args.force:
            print("  Skipping: calibration already exists")
            skip_count += 1
            continue

        try:
            success, quit_all = process_scan(subject_id, session, ct_path, args.output, args.force)
            if quit_all:
                print("\n  User requested to quit - exiting application")
                break
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  Error: {e}")
            fail_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total scans: {len(scans)}")
    print(f"Successful: {success_count}")
    print(f"Failed/Aborted: {fail_count}")
    print(f"Skipped (existing): {skip_count}")


if __name__ == "__main__":
    main()
