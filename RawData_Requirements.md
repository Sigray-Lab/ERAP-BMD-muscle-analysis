# Raw Data Requirements

This document describes the expected raw input data for the ERAP BMD & Muscle Composition Analysis Pipeline.

## Overview

The pipeline processes **CT scans** from 13 subjects (sub-101 through sub-114, excluding sub-106) at two timepoints (Baseline and Followup). Each scan was acquired on a **GE Discovery MI** PET-CT system at Karolinska Institutet using a helical acquisition with STANDARD reconstruction kernel.

All scans include a **Mindways QCT calibration phantom** in the field of view, positioned under the patient at the L1-L2 vertebral level.

Raw imaging data are not included in this repository due to patient privacy regulations.

## Expected Directory Structure

```
RawData/bmd_ct/
├── sub-101/
│   ├── ses-Baseline/
│   │   └── ct/
│   │       ├── sub-101_ses-Baseline_desc-BMD_rec-stnd1.25mm_ct.nii.gz
│   │       ├── sub-101_ses-Baseline_desc-BMD_rec-stnd1.25mm_ct.json
│   │       ├── sub-101_ses-Baseline_desc-BMD_rec-stnd2.5mm_ct.nii.gz
│   │       └── sub-101_ses-Baseline_desc-BMD_rec-stnd2.5mm_ct.json
│   └── ses-Followup/
│       └── ct/
│           ├── sub-101_ses-Followup_desc-BMD_rec-stnd1.25mm_ct.nii.gz
│           ├── sub-101_ses-Followup_desc-BMD_rec-stnd1.25mm_ct.json
│           ├── sub-101_ses-Followup_desc-BMD_rec-stnd2.5mm_ct.nii.gz
│           └── sub-101_ses-Followup_desc-BMD_rec-stnd2.5mm_ct.json
├── sub-102/
│   └── ...
└── sub-114/
    └── ...
```

The pipeline expects the `--data` argument to point to the `RawData/bmd_ct/` directory (or equivalent path).

## BIDS Naming Convention

Files follow the [Brain Imaging Data Structure](https://bids.neuroimaging.io/) naming convention:

```
{subject}_{session}_{description}_{reconstruction}_{modality}.{extension}
```

| Component | Format | Example |
|-----------|--------|---------|
| Subject | `sub-XXX` | `sub-101` |
| Session | `ses-Baseline` or `ses-Followup` | `ses-Baseline` |
| Description | `desc-BMD` | `desc-BMD` |
| Reconstruction | `rec-stnd1.25mm` or `rec-stnd2.5mm` | `rec-stnd1.25mm` |
| Modality | `ct` | `ct` |
| Extension | `.nii.gz` (image) or `.json` (metadata) | `.nii.gz` |

## File Inventory

### Per Session (4 files)

| File | Format | Size | Used by Pipeline |
|------|--------|------|------------------|
| `*_rec-stnd1.25mm_ct.nii.gz` | NIfTI (gzip) | ~30-32 MB | **Yes** (primary input) |
| `*_rec-stnd1.25mm_ct.json` | JSON sidecar | ~900 B | No (DICOM metadata reference) |
| `*_rec-stnd2.5mm_ct.nii.gz` | NIfTI (gzip) | ~15 MB | No (lower resolution) |
| `*_rec-stnd2.5mm_ct.json` | JSON sidecar | ~900 B | No (DICOM metadata reference) |

### Pipeline Glob Pattern

The pipeline selects CT files using this pattern (from `run_pipeline.py`):

```python
ct_files = list(session_dir.glob("*_rec-stnd1.25mm_ct.nii.gz"))
```

Only the **1.25 mm** reconstruction is used.

## Subject Inventory

| Subject | ses-Baseline | ses-Followup |
|---------|:---:|:---:|
| sub-101 | ✓ | ✓ |
| sub-102 | ✓ | ✓ |
| sub-103 | ✓ | ✓ |
| sub-104 | ✓ | ✓ |
| sub-105 | ✓ | ✓ |
| sub-107 | ✓ | ✓ |
| sub-108 | ✓ | ✓ |
| sub-109 | ✓ | ✓ |
| sub-110 | ✓ | ✓ |
| sub-111 | ✓ | ✓ |
| sub-112 | ✓ | ✓ |
| sub-113 | ✓ | ✓ |
| sub-114 | ✓ | ✓ |

**Total:** 13 subjects × 2 sessions = 26 scans (sub-106 not enrolled)

## Acquisition Parameters

| Parameter | Value |
|-----------|-------|
| Scanner | GE Discovery MI (PET-CT) |
| Institution | Karolinska Institutet |
| Scan mode | Helical |
| Reconstruction kernel | STANDARD |
| Slice thickness | 1.25 mm (primary) / 2.5 mm (secondary) |
| Anatomical coverage | L1-L2 vertebral region |
| Calibration phantom | Mindways QCT (in FOV) |
| DICOM conversion | dcm2niix v1.0.20220720 |

## JSON Sidecar Contents

Each `.nii.gz` file has a matching `.json` sidecar containing DICOM metadata:

| Field | Description |
|-------|-------------|
| `Modality` | `CT` |
| `Manufacturer` | `GE` |
| `ManufacturersModelName` | `Discovery MI` |
| `SeriesDescription` | `CT_BMD_Stnd-1.25mm` or `CT_BMD_Stnd-2.5mm` |
| `ConvolutionKernel` | `STANDARD` |
| `ScanOptions` | `HELICAL MODE` |
| `ImageType` | `["ORIGINAL", "PRIMARY", "AXIAL"]` |
| `ConversionSoftware` | `dcm2niix` |
