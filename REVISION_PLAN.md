# Revision plan after the adversarial review (2026-09-05)

Author: Claude (Fable 5.1), written after independently re-checking the review against the code,
the derived data and the raw scans. This is a plan, not an executed change. Nothing in the project
has been modified yet.

## 0. Independent assessment of the review

I did not take the review at face value. What I verified myself:

| Claim | My check | Verdict |
|---|---|---|
| F01 body isolation keeps posterior elements | Read `01_segmentation.py:270-300`: computes anterior centroids, never uses them, keeps largest 2D then largest 3D component. Overlays for sub-101/113/114 show the whole arch inside the "body" and 5 mm-core islands inside pedicles. | **Confirmed. Real defect.** |
| F02 phantom HU varies along z | Re-read the *untracked* fixed-xy profile for sub-103 Baseline: 200-rod swings 208 to 247 HU with troughs exactly at the two vertebral-body levels; regression slope varies 0.76 to 0.99 mg/cm3 per HU within one scan. Across 52 vertebra-sessions the slope at the ROI differs from the slope at z=50 by up to 11 %. | **Confirmed and, in my view, under-rated by the reviewer.** This is classic beam hardening from the vertebral body itself. Sampling the phantom at the same slices as the vertebra is the standard QCT practice (Cann-Genant) and this pipeline does not do it. |
| F03 fractional-HU classifier gaps | `THRESHOLDS` are closed integer intervals; drift offset is fractional. | Confirmed. Small effect, trivial fix. |
| F06 3 mm vs 5 mm | `DEFAULT_EROSION_MM = 5.0`; Methods and CLAUDE.md say 3 mm; README says 5 mm. | Confirmed. Text fix only. |
| F07 manual calibration described as Hough | All 26 files say `manual_clicker`; residual warning is 15 HU, docs say 10. | Confirmed. Text fix only. |
| F08 ball(7), symmetry ignores envelope | `ball(int(mean(5/zooms)))` = 7 voxels; `compute_symmetry_index` never uses its `envelope_mask` argument. | Confirmed. Text fix plus one-line code fix. |
| F14 kota1/kota2 reversed | Rendered the 2.5 mm volume at index 19 and 45 for sub-109 and compared with the physicist's screenshots: slice 20/65 is unambiguously the inferior vertebra (aortic calcification, bulbous spinous process match). ImageJ does not flip z. | **Confirmed for the screenshot-backed subjects.** The 4 without screenshots need the physicist. |
| L01 rerunner drops `subject_id` | `rerun_analysis.py:69` calls `detect_central_vertebrae(seg_dir)`. | Confirmed. Becomes moot once the override is removed (see A1). |
| F04 absolute level unverified | Reviewer could not anchor; registration confirms the *same* pair in both sessions (Dice 0.84-0.94). | Agree it is unverified. Disagree that it is a pipeline defect: for a paired design the same-pair result is what matters. It is a manuscript-wording and clinical-reading problem. |
| F05 phantom identity | The 0/50/100/200 and fat -100 rod set does not match Mindways Model 3 (K2HPO4, different nominal values). The physicist's own slopes agree with ours, so both used the same nominal numbers. | Agree it must be traced. Cannot be solved in code. |

Where I push back on the review:

1. **Its corrected tables are evidence, not results.** They come from review-side scripts with one
   hand-patched merged instance (sub-111). The project's own pipeline must be changed and re-run so
   that every number in the manuscript traces to `Scripts/`.
2. **F02 is the largest numeric issue, not F01.** Body-only masks move the primary change from
   +2.11 to +2.31 mg/cm3. Co-located calibration moves it to +0.46. The reviewer filed F02 as
   "sensitivity only" because no certified correction exists; I think the standard-practice argument
   is strong enough to make co-located calibration the primary method.
3. **Do not switch to 3 mm erosion.** Keep 5 mm, fix the text. The reviewer agrees; stated here so it
   is not re-litigated.
4. **Do not implement a physical 5 mm ellipsoid for the envelope.** Document ball(7) accurately.
   Re-implementing costs a change to explain with no scientific gain.
5. **Do not fix the Hough path; delete it.** It was never used, it has an axis bug, and it is the
   reason the Methods text is wrong. Dead code that misdescribes the method is worse than no code.
6. **mA heterogeneity, noise matching, median vs mean, weighting:** the reviewer's own conclusion is
   that these show no defect. They go into a limitations sentence and a supplementary sensitivity
   table, nothing more.
7. **sub-106 in the TSV, JSON NaN, first-glob selection, directory flags:** cosmetic. Fix in one
   small commit, do not let them consume attention.

## 1. Tier A: result-changing fixes (require a full analysis re-run)

TotalSegmentator does **not** need to be re-run for the existing tasks. One new task is needed
(`--task vertebrae_body`, ~1-2 min per scan on MPS, the reviewer already ran it into
`Review_adversarial/sessions/*/body_reference/`, which can be reused after QC or regenerated into
`DerivedData` for provenance).

### A1. Body-only vertebral isolation (F01, also resolves F06's "anterior component" wording, L01, L04, sub-114 Z-split)

Replace `isolate_vertebral_body` with:

1. Run `TotalSegmentator --task vertebrae_body` per scan into `DerivedData/.../segmentations/`.
2. Connected components of the body mask. Discard components touching the z-edge of the FOV
   (partial bodies) and components < 2000 voxels.
3. Assign each body component to a TotalSegmentator vertebra label by majority overlap; if one
   body component overlaps two labels substantially (sub-111 Baseline case), split it by marker
   watershed seeded from the 1.5 mm-eroded interiors, exactly as the reviewer did. Log every split.
4. Select the two largest complete bodies, order by z-centroid: superior = "upper", inferior = "lower".
   This removes the need for `SUBJECTS_REQUIRING_Z_SPLIT` and the subject-ID override; delete both.
5. Endplate exclusion (10 %) from the **body** z-extent, then 5 mm EDT erosion as now.
6. Muscle slab z-range = union of the two body z-extents (this propagates to all muscle metrics,
   as the reviewer's `table1_corrected.csv` already did).

QC: a per-scan overlay (axial through each body plus sagittal) with the old and new body and
core contours, all 52 reviewed visually before numbers are accepted. Pay special attention to
sub-108, sub-113 Baseline (T11 fragment) and sub-114 (holes).

Expected effect: primary change ≈ +2.3 mg/cm3, muscle volume p ≈ 0.05, CSA p ≈ 0.07. Core
volumes become consistent between sessions.

### A2. Co-located phantom calibration (F02) and co-located drift offset

Keep the manual clicks (they are correct, all 130 verified inside the rods) but change what is
sampled:

1. For every scan build a per-slice rod profile over the whole z-range using the clicked (x, y)
   centres, re-centred per slice by phase correlation of the phantom patch (max shift found by the
   reviewer is 4.1 voxels, so tracking is needed with a 4 mm ROI). Save the profile and a QC plot
   per scan.
2. For each vertebra, fit the regression on rod means averaged over **that vertebra's trabecular
   ROI slices** (typically 30-40 slices, far less noisy than a 9-slice window). Store
   `calibration_bmd.json` with `global_z50` (the current numbers, kept for the sensitivity table)
   and `per_vertebra` entries.
3. For muscle, the drift offset becomes the base-rod mean over the muscle-slab z-range instead of
   the 9 slices at z=50 (the base rod varies by ~20 HU along z in sub-103; a threshold-based
   metric should not depend on where the clicks happened to be).
4. Bone analysis reads the per-vertebra calibration. Muscle analysis reads the slab offset.

Expected effect: primary change drops to roughly +0.5 to +0.7 mg/cm3 (combined with A1). Both
remain non-significant; the point is that the number is defensible.

Methods sentence to add: "Because rod attenuation varied systematically along z within a scan
(beam hardening from the vertebral bodies), calibration was performed per vertebra using rod
values averaged over the same slices as the trabecular ROI."

### A3. Continuous half-open HU classes (F03)

`THRESHOLDS`: IMAT [-190, -30), low-density [-30, 30), normal [30, 150], muscle_all [-30, 150].
Document in code and Methods. Effect on the two fraction rows is < 0.1 percentage points.

### A4. Symmetry index within the envelope (F08, QC only)

Use the envelope argument that is already passed. No manuscript number depends on it.

### A5. One statistics script producing everything (F11, F12)

Extend `07_statistics_summary.py` to output mean (SD), Δ with 95 % CI, Cohen's dz, p, and the
ancillary Results numbers (trabecular volumes, Dice mean/range/count, calibration R2 range, drift
range, symmetry range). Table 1 in the manuscript is pasted from this file and nowhere else.
Also write a `sensitivity_table.csv` with: original method, A1 only, A1+A2, 3 mm erosion,
no HU filter, median. That becomes a supplementary table.

### A6. Provenance manifest (F16, L02)

Every module writes a `manifest.json` next to its outputs: command, package versions,
TotalSegmentator version and task, input file hash, git commit, timestamp. The existence check
in `01_segmentation.py` must validate both muscles, both target vertebrae, non-empty content and
the manifest before skipping.

## 2. Tier B: record corrections (no numbers change, must be done before submission)

| Item | Where | Fix |
|---|---|---|
| Erosion 3 mm | Methods, CLAUDE.md | 5 mm |
| Hough detection | Methods, README, CLAUDE.md | "Rod centres were placed manually on a mid-phantom slice; rod means were sampled in 4 mm-radius cylinders"; delete Hough code |
| Residual threshold | CLAUDE.md | 15 HU (or change code to 10, either is fine; pick one) |
| "5 mm spherical structuring element" | Methods, README | "7-voxel ball (≈4.8 mm in-plane, 4.4 mm axial)" |
| "Swiss cheese" justification | Methods, README | Soften: masks contain 0.3-10 % HU-fat; envelope is used to include IMAT at the fascial boundary |
| "Muscle mass" | README | "muscle volume" |
| VAT/SAT "1/13 adequate FOV" | Methods, CLAUDE.md | Re-run adipose module (F09); state the real reason (SAT truncated in 15/26; SAT overlaps phantom; VAT FOV never assessed); keep not reporting |
| Trabecular volumes 7.8/9.9 | Results | Regenerate from A5 |
| Dice mean .54, "<2 cm3" explanation | Results, CLAUDE.md | Regenerate; drop the blanket explanation |
| Slice thickness 1.25 mm | Methods, RawData_Requirements | "1.25 mm reconstruction, 0.625 mm slice spacing" |
| README data layout | README | Real filenames (`*_rec-stnd1.25mm_ct.nii.gz`, `muscle_compartment.nii.gz` at session root) |
| sub-106 in `ct_bmd_files.tsv` | RawData | Remove rows or annotate as not enrolled |
| Stale `statistics_summary.*`, `validation_results.csv` | Outputs | Regenerated by A5; delete old ones |
| Rapamycin correlation | Outputs | Add `Scripts/09_rapa_correlation.py` with the exclusion list and its source; regenerate figure |
| Manual validation mapping | `08_manual_bmd_validation.py` | Per-subject mapping file `manual_kota_mapping.csv` (9 confirmed as kota1 = inferior; 4 pending). Report the ordering-invariant two-vertebra change agreement as the headline (R2 0.79, bias +0.6, RMSE 3.2 mg/cm3 at n=13) and level-specific agreement for the confirmed 9 |
| Vertebral naming | Manuscript throughout | Until Tier C2 is done: "the two adjacent vertebral bodies centred in the field of view (L1-L2 by protocol)" with the per-subject label table as a supplement |

## 3. Tier C: actions only you can take (nothing in code can substitute)

1. **Phantom certificate (F05).** Ask medical physics (the physicist who did the manual measurements) which phantom was under the
   patient (manufacturer, model, serial) and for the certificate with the rod densities and units.
   The 0/50/100/200 + fat -100 set does not match Mindways Model 3. If the certificate differs,
   the calibration constants change and all absolute vBMD values rescale (paired p-values survive
   a pure scale factor, not an intercept change).
2. **Vertebral level reading (F04).** Have a radiologist or experienced reader identify the level
   of the two measured bodies in each subject using the wide-FOV PET/CT bed-position CT (count
   from the last rib-bearing vertebra). I can prepare a 13-panel sagittal montage with the two
   bodies outlined to make this a 20-minute job. Until then the manuscript uses the neutral wording
   above.
3. **kota mapping for sub-103, 104, 110, 112.** One e-mail to the physicist asking which slice
   range kota1 and kota2 covered, or confirmation that kota1 was always the caudal vertebra.
4. **Slice thickness.** Locate the original BMD-CT DICOM (or the scanner protocol sheet) to
   confirm 1.25 mm thickness with 0.625 mm spacing.

## 4. Tier D: latent bugs, one small commit

- L06: `allow_nan=False` and convert non-finite to null in all JSON writers.
- L07: `check_sat_fov_adequate` returns "unknown" on an empty mask; remove `body_trunc` as a VAT synonym.
- L08: assert exactly one CT match per session; `has_baseline/has_followup` require successful bone and muscle results, not directory existence.
- L05: `analyze_vertebra` returns the mask it actually measured; save that mask and its effective erosion.
- L03: deleted with the Hough path.

## 5. Execution order

1. Tier D + Tier B code-side items (delete Hough, thresholds, symmetry, JSON, discovery). Commit.
2. A1 body-only isolation, run on all 26, visual QC of 52 overlays. Commit with QC images.
3. A2 co-located calibration, run on all 26, QC plots of rod profiles. Commit.
4. A3-A6, full analysis re-run via `rerun_analysis.py` (fixed), aggregation, statistics, sensitivity table. Commit; tag as `v2-review-response`.
5. Update Methods/Results/README/CLAUDE.md from the regenerated numbers. Push to GitHub.
6. Send the three Tier C requests in parallel with step 1; fold the answers in when they arrive
   (phantom constants are a one-line change in `manual_calibration.py` followed by re-running step 4).

Estimated effort: steps 1-4 about two working days of pipeline work plus QC review time; step 5
half a day.

## 6. Acceptance criteria before the manuscript is updated

- All 52 body masks visually accepted; no core voxel posterior to the body's posterior wall.
- Core volume differs by < 15 % between sessions of the same vertebra unless a QC note explains it.
- Per-vertebra calibration R2 ≥ 0.995 in all 52 fits; rod-profile QC plots show no tracking loss.
- Every number in Table 1 and the Results paragraphs is produced by `07_statistics_summary.py` from one `results.csv`, and `results.csv` carries the git commit hash that produced it.
- `rerun_analysis.py` on any single session reproduces that session's JSON bit-for-bit.
- README example commands run as written on this machine.

## 7. PI decisions (2026-09-05, in chat)

- A1 body-only isolation: approved. **Keep the sub-114 Z-split override and the zsplit files in place
  until the body-only path has been run on all 26 scans and its sub-114 masks visually confirmed.**
  Only then is the override retired. Same for `rerun_analysis.py`: fix the missing `subject_id`
  argument now (L01) so the override keeps working during the transition.
- A2 co-located calibration: **not yet agreed.** PI wants the rod average to stay the calibration
  basis; open question is over which slices to average. Options on the table: (a) 9 slices at
  z=50 as now, (b) all slices, (c) the slices of each vertebra's core. Decide before implementing.
- F14 kota mapping: approved, fix in the validation script with a per-subject mapping file.
- Housekeeping done before any pipeline edit: stale summaries and L3 leftovers moved to `old/`,
  revert point tagged `v1-as-reviewed` and pushed to GitHub.
