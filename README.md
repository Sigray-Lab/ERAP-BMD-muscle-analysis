# ERAP BMD & Muscle Composition Analysis Pipeline

Analysis pipeline for quantifying trabecular bone mineral density and paraspinal muscle composition from spine CT in the **ERAP clinical trial** (a pilot study of rapamycin in early-stage Alzheimer's disease).

## Background

Rapamycin inhibits mTOR, a central regulator of bone metabolism, muscle protein synthesis, autophagy and adipogenesis. This pipeline measures whether rapamycin treatment changes trabecular vBMD and paraspinal muscle quality/size between pre- and post-treatment CT scans of two adjacent lumbar vertebrae (L1–L2 by protocol).

Each scan includes a density calibration phantom under the patient with inserts of nominal 0, 50, 100 and 200 mg/cm³ (plus a −100 fat-equivalent insert used only for QC), enabling HU → BMD conversion per scan.

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_segmentation.py` | TotalSegmentator (`roi_subset` T10–L4 + autochthon; `tissue_4_types`), vertebra selection, body-only isolation, muscle envelope |
| 1b | `01b_segment_vertebral_bodies.py` | TotalSegmentator `--task vertebrae_body` (vertebral bodies without the posterior arch), with a manifest per scan |
| 2 | `manual_calibration.py` | Interactive rod clicker: one click per rod on one slice; 4 mm-radius cylinders over 9 slices. **This is how all 26 scans were calibrated.** |
| 2b | `02_phantom_zprofile.py` | Follows the phantom tray from the clicks along z, samples every rod on every slice, writes strict QC (per-scan sheets and cohort contact sheets) |
| 2c | `02_phantom_calibration.py` | Loader for the manual calibration files (no automatic detection exists any more) |
| 3 | `03_bone_analysis.py` | 5 mm distance-transform erosion, −50…400 HU, **per-vertebra co-located calibration** (rod means averaged over the ROI slices) |
| 4 | `04_muscle_analysis.py` | SMD, IMAT, low-density fraction, CSA, volume; continuous half-open HU classes; co-located drift offset |
| 5 | `05_adipose_analysis.py` | VAT/SAT from `tissue_4_types` (not reported at group level: SAT truncated by the FOV and overlapping the phantom) |
| 6 | `06_results_aggregation.py` | Per-session JSON → `Outputs/results.csv` with change metrics and a provenance manifest |
| 7 | `07_tissue_validation.py` | IMAT: HU-based vs TotalSegmentator `intermuscular_fat`, both restricted to the muscle envelope (Dice) |
| 8 | `07_statistics_summary.py` | Table 1 (Δ, 95 % CI, dz, p), ancillary numbers, sensitivity table, figures |
| 9 | `08_manual_bmd_validation.py` | Pipeline vs physicist manual vBMD (per-subject kota mapping file) |
| 10 | `09_rapa_correlation.py` | Rapamycin concentration vs Δ outcome (documented exclusions) |
| – | `run_pipeline.py` | Orchestrator (steps 1–7 per scan, then aggregation) |
| – | `rerun_analysis.py` | Re-run steps 3–7 from existing segmentations and calibration; writes `analysis_manifest.json` per session |

### Quantitative metrics

**Bone (two target vertebral bodies, "L1" = superior, "L2" = inferior):**

| Metric | Unit | Description |
|--------|------|-------------|
| `L1L2_vBMD_mean_mgcm3` | mg/cm³ | Primary endpoint: mean of the two trabecular vBMD values |
| `L1_vBMD_mean_mgcm3`, `L2_vBMD_mean_mgcm3` | mg/cm³ | Per vertebra |
| `L1_trabecular_volume_cm3` | cm³ | Eroded trabecular ROI volume |
| `*_vBMD_mean_mgcm3_z50_calibration` | mg/cm³ | Same ROI with the click-slice calibration (sensitivity) |

**Muscle (erector spinae compartment over the z-range of the two bodies):**

| Metric | Unit | Description |
|--------|------|-------------|
| `muscle_SMD_mean_hu` | HU | Skeletal muscle density (lower = more fat infiltration) |
| `muscle_low_density_percent` | % | Low-density muscle [−30, 30) HU as % of muscle [−30, 150] HU |
| `IMAT_percent` | % | Intermuscular adipose [−190, −30) HU as % of the envelope |
| `muscle_tissue_volume_cm3` | cm³ | Muscle volume (= mean CSA × slab height) |
| `muscle_CSA_mean_cm2` | cm² | Mean muscle cross-sectional area |

## Quick Start

### Prerequisites

```bash
pip install nibabel numpy pandas scipy scikit-image matplotlib PyYAML
pip install TotalSegmentator      # 2.12 used here; tissue_4_types needs an academic licence:
totalseg_set_license -l <your_license_key>
```

### Running

```bash
# 1. segmentation + analysis for all subjects (manual calibration must exist per session)
python Scripts/run_pipeline.py --data ../RawData/bmd_ct --output .

# 2. manual phantom calibration (interactive; once per scan) — run BEFORE step 1 for new scans
python Scripts/manual_calibration.py --data ../RawData/bmd_ct --output .

# re-run the analysis only (existing segmentations, clicks and phantom_zprofile.json)
python Scripts/rerun_analysis.py --all
python Scripts/rerun_analysis.py --subject sub-101 --session ses-Baseline

# statistics, tables and figures
python Scripts/07_statistics_summary.py
```

On macOS the `vertebrae_body` task runs through a wrapper that keeps temporary paths short (AF_UNIX socket length limit); it uses a git-ignored `tmp_ts/` in the project root.

## Data Requirements

```
RawData/bmd_ct/
└── sub-XXX/
    └── ses-Baseline | ses-Followup/
        └── ct/
            ├── sub-XXX_ses-YYY_desc-BMD_rec-stnd1.25mm_ct.nii.gz   # used (0.68 x 0.68 x 0.625 mm, 512x512x129)
            ├── sub-XXX_ses-YYY_desc-BMD_rec-stnd1.25mm_ct.json
            ├── sub-XXX_ses-YYY_desc-BMD_rec-stnd2.5mm_ct.nii.gz    # not used by the pipeline
            └── sub-XXX_ses-YYY_desc-BMD_rec-stnd2.5mm_ct.json
```

Exactly one `*_rec-stnd1.25mm_ct.nii.gz` per session is required; ambiguity is an error.

## Output Structure

```
DerivedData/sub-XXX/ses-YYY/
├── segmentations/                  # TotalSegmentator roi_subset + tissue_4_types
│   └── vertebrae_body/             # body-only mask, invocation.log, manifest.json
├── vertebral_bodies/
│   ├── L1_body.nii.gz, L2_body.nii.gz      # body-only, endplates excluded
│   ├── trabecular_masks/L{1,2}_trabecular.nii.gz   # the ROI actually measured
│   ├── vertebra_detection.json     # which label became L1/L2 (+ subject overrides)
│   └── body_isolation.json         # instance assignment, splits, warnings
├── muscle_compartment.nii.gz       # envelope
├── muscle_masks/                   # envelope, imat, low_density, normal, all, classification
├── phantom_calibration.json, calibration_bmd.json, calibration_hu_stability.json   # manual clicks
├── phantom_zprofile.json           # per-slice tracked rod means, QC flags
├── bone_results.json, muscle_results.json, adipose_results.json, validation_results.json
└── analysis_manifest.json          # git commit, package versions, methods used
QC/sub-XXX/ses-YYY/                 # phantom_tracking.png, body_isolation_v2.png, envelope, classification, ...
Outputs/                            # results.csv, table1.*, statistics_summary.csv, sensitivity_table.*,
                                    # ancillary_numbers.*, Figures/, validation_results.csv — see Outputs/README_outputs.md
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Vertebral body from the `vertebrae_body` model, assigned to the selected vertebra by overlap | The whole-vertebra label's largest component includes the posterior arch in pedicle slices; the body model does not |
| 10 % endplate exclusion on the body z-extent, then 5 mm distance-transform erosion | Excludes endplates and the cortical shell; trabecular ROI 6–17 cm³ |
| Manual rod clicks, then tracked sampling of every rod on every slice | Automatic rod detection was unreliable; the tray can shift up to 2.9 mm over the scan; QC sheets show the circle at both ends of every scan |
| Per-vertebra calibration from rod means over the ROI slices | Rod HU varies ~15 % along z with beam hardening from the vertebral bodies; sampling the phantom in the same slices as the bone is standard QCT practice and agreed better with independent manual measurements |
| Muscle envelope: closing with a 7-voxel ball (≈4.8 mm in-plane, 4.4 mm axial) + 2D hole fill | Includes intermuscular fat at the fascial boundary |
| Continuous half-open HU classes | The drift offset is fractional; closed integer intervals left voxels unclassified |
| Two largest complete vertebrae, named by z position; `sub-114` Z-split override retained | TotalSegmentator labels are unreliable in an 81 mm FOV; the override was kept until the body-only path is validated on that subject |
| VAT/SAT not reported | SAT is truncated by the FOV in most scans and overlaps the phantom; VAT FOV adequacy was never assessed |

## Known Limitations

- **Absolute vertebral level is not verified.** The two measured bodies are the same physical pair at both visits (registration Dice 0.84–0.94), but whether they are L1–L2 in every subject requires reading a wide-FOV CT; the TotalSegmentator label mapping is in `Outputs/vertebra_label_mapping.csv`.
- **Phantom certificate.** The nominal rod densities (0/50/100/200) are program constants; the actual phantom model and certificate must be confirmed.
- **Slice thickness.** The 1.25 mm reconstruction has 0.625 mm slice spacing; nominal thickness is not stored in the sidecars.

## Development

Developed with Claude Code (Anthropic); maintained by the [Sigray Lab](https://github.com/Sigray-Lab) at Karolinska Institutet. An adversarial technical review (2026-09) led to the body-only isolation, co-located calibration and the QC material above; see `REVISION_PLAN.md`.

## References

1. Wasserthal et al., "TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images," *Radiology: AI*, 2023
2. Cann CE, Genant HK. Precise measurement of vertebral mineral content using computed tomography. *J Comput Assist Tomogr* 1980
3. American College of Radiology, "ACR–SPR–SSR Practice Parameter for the Performance of Musculoskeletal Quantitative Computed Tomography (QCT)"

## License

Part of the ERAP clinical trial. Raw imaging data are not included in this repository.
