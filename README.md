# ERAP BMD & Muscle Composition Analysis Pipeline

Analysis pipeline for quantifying bone mineral density, muscle composition, and adipose tissue from CT imaging data in the **ERAP clinical trial** — a Phase IIa study evaluating rapamycin treatment in early-stage Alzheimer's disease.

## Background

Rapamycin inhibits mTOR, a central regulator of bone metabolism, muscle protein synthesis, autophagy, and adipogenesis. This pipeline measures whether rapamycin treatment affects trabecular bone density, muscle quality, and visceral fat using pre- and post-treatment CT scans of the L1-L2 vertebral region.

Each scan includes a **density calibration phantom** (Mindways QCT) in the field of view with known-density inserts, enabling quantitative HU-to-BMD conversion.

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_segmentation.py` | TotalSegmentator segmentation, vertebral body isolation, muscle envelope creation |
| 2 | `02_phantom_calibration.py` | Automatic phantom rod detection and HU-to-BMD calibration |
| 3 | `03_bone_analysis.py` | Trabecular vBMD from eroded vertebral bodies using phantom calibration |
| 4 | `04_muscle_analysis.py` | Muscle density, IMAT, and myosteatosis from muscle compartment envelope |
| 5 | `05_adipose_analysis.py` | VAT/SAT volumes from TotalSegmentator anatomical labels |
| 6 | `06_results_aggregation.py` | Aggregate per-session JSONs into a single CSV with change metrics |
| 7 | `07_tissue_validation.py` | Validate HU-based IMAT against TotalSegmentator tissue_4_types |
| 8 | `07_statistics_summary.py` | Paired t-tests for all primary outcomes |
| - | `run_pipeline.py` | Main orchestrator (runs steps 1-7) |
| - | `rerun_analysis.py` | Re-run analysis without TotalSegmentator (uses existing segmentations) |

### Quantitative Metrics

**Bone (L1-L2 vertebral bodies):**

| Metric | Unit | Description |
|--------|------|-------------|
| `L1L2_vBMD_mean_mgcm3` | mg/cm3 | Primary endpoint — phantom-calibrated trabecular BMD |
| `L1_vBMD_mean_mgcm3` | mg/cm3 | Individual vertebra BMD |
| `L1_trabecular_volume_cm3` | cm3 | Eroded trabecular region volume |

**Muscle (erector spinae at L1-L2 level):**

| Metric | Unit | Description |
|--------|------|-------------|
| `muscle_SMD_mean_hu` | HU | Skeletal muscle density (lower = more fat infiltration) |
| `muscle_low_density_percent` | % | Myosteatosis burden (-29 to +29 HU) |
| `IMAT_percent` | % | Intermuscular adipose tissue (-190 to -30 HU) |
| `muscle_tissue_volume_cm3` | cm3 | Muscle mass |

**Adipose:**

| Metric | Unit | Description |
|--------|------|-------------|
| `VAT_volume_cm3` | cm3 | Visceral adipose tissue (TotalSegmentator labels) |
| `muscle_CSA_VAT_ratio` | ratio | Sarcopenic obesity index |

## Quick Start

### Prerequisites

```bash
pip install nibabel numpy pandas scipy scikit-image matplotlib opencv-python PyYAML
pip install TotalSegmentator

# tissue_4_types task requires a TotalSegmentator academic license:
totalseg_set_license -l <your_license_key>
```

### Running the Pipeline

```bash
# Full pipeline on all subjects (skips TotalSegmentator if output exists)
python Scripts/run_pipeline.py --data ../RawData/bmd_ct --output .

# Force re-run TotalSegmentator
python Scripts/run_pipeline.py --data ../RawData/bmd_ct --output . --force

# Single subject
python Scripts/run_pipeline.py --data ../RawData/bmd_ct --output . --subject sub-101

# Re-run analysis only (uses existing segmentations)
python Scripts/rerun_analysis.py --subject sub-101 --session ses-Baseline
```

## Data Requirements

The pipeline expects BIDS-like input data (not included in this repository):

```
RawData/bmd_ct/
└── sub-XXX/
    ├── ses-Baseline/
    │   └── ct.nii.gz       # CT scan with phantom in FOV
    └── ses-Followup/
        └── ct.nii.gz
```

## Output Structure

```
DerivedData/
└── sub-XXX/ses-YYY/
    ├── segmentations/              # TotalSegmentator output
    ├── vertebral_bodies/
    │   ├── L1_body.nii.gz          # Isolated vertebral body
    │   ├── L2_body.nii.gz
    │   └── vertebra_detection.json # Original → L1/L2 mapping QC
    ├── muscle_masks/
    │   └── muscle_compartment.nii.gz
    ├── bone_results.json
    ├── muscle_results.json
    ├── adipose_results.json
    ├── calibration_bmd.json        # Phantom HU→BMD regression
    └── calibration_hu_stability.json
Outputs/
└── results.csv                     # Aggregated results (all subjects)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Vertebral body isolation (anterior component only) | Excludes posterior elements that bias trabecular BMD |
| 5 mm distance-transform erosion | Isolates trabecular bone, excludes cortical shell |
| Morphological closing + 2D hole filling for muscle envelope | Solves "Swiss cheese" problem — TotalSegmentator excludes fat voxels from muscle |
| Phantom-based HU→BMD calibration per scan | Corrects for scanner drift between timepoints |
| Robust vertebra detection (2 largest by voxel count, assigned by Z-position) | Handles TotalSegmentator mislabeling in limited-FOV scans |
| Z-split override for sub-114 | TotalSegmentator produced overlapping L1/L2 masks spanning multiple vertebrae; masks are combined and split at centroid midpoint |
| IMAT validation restricted to muscle envelope | Prevents phantom calibration rods from being counted as fat |
| VAT from anatomical labels (not HU thresholding) | More robust boundaries; phantom artifact-immune |

## Known Issues

- **TotalSegmentator vertebra mislabeling**: With spine-focused FOV, vertebrae are often mislabeled (e.g., L1 called "T12"). The pipeline handles this by selecting the 2 largest vertebrae and assigning by Z-position.
- **Sub-114 overlapping masks**: TotalSegmentator produces masks spanning multiple vertebrae. Handled via Z-split approach (combine + split at midpoint).
- **SAT truncation**: Spine-focused FOV often cuts off lateral subcutaneous fat. SAT is only reported when FOV is adequate.
- **tissue_4_types phantom artifacts**: Narrow FOV causes TotalSegmentator to mislabel phantom rods as adipose. VAT/SAT from tissue_4_types is disabled; IMAT comparison is restricted to muscle envelope.

## Development

This pipeline was developed collaboratively using [Claude Code](https://claude.ai/code) (Anthropic) and is maintained by the [Sigray Lab](https://github.com/Sigray-Lab) at Karolinska Institutet.

See also the companion pipeline for periodontal FDG-PET analysis from the same trial: [ERAP-periodontal-analysis](https://github.com/Sigray-Lab/ERAP-periodontal-analysis).

## References

1. Wasserthal et al., "TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images," *Radiology: AI*, 2023
2. American College of Radiology, "ACR-SPR-SSR Practice Parameter for the Performance of Musculoskeletal Quantitative Computed Tomography (QCT)"

## License

This project is part of the ERAP clinical trial. Raw imaging data are not included in this repository due to patient privacy regulations.
