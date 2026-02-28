# CT-Derived Bone, Muscle, and Body Composition Outcomes — Methods and Results

## Methods

### Study Design and Participants

Thirteen participants from the ERAP clinical trial (a pilot study evaluating rapamycin in early-stage Alzheimer's disease) underwent CT imaging at two timepoints: baseline and follow-up. Each participant was scanned on a GE Discovery MI PET-CT system (Karolinska Institutet, Stockholm, Sweden) using a helical acquisition with standard reconstruction kernel at 1.25 mm slice thickness, covering the L1–L2 vertebral region. A Mindways QCT calibration phantom containing inserts of known density (0, 50, 100, and 200 mg/cm3 hydroxyapatite equivalent) was positioned in the field of view beneath the patient for all scans.

### Image Processing Pipeline

All CT images were processed using an automated pipeline built in Python (source code available at https://github.com/Sigray-Lab/ERAP-BMD-muscle-analysis). Anatomical segmentation was performed using TotalSegmentator (v2, Wasserthal et al., 2023), which provided vertebral body and paraspinal muscle labels (T10–L4, bilateral autochthonous muscles).

#### Vertebra Identification

Because the spine-focused field of view sometimes caused TotalSegmentator to mislabel vertebral levels, a robust detection algorithm was applied: all detected vertebral masks (T10–L4) were ranked by voxel count, the two largest (assumed fully within the field of view) were selected, and L1/L2 identity was assigned by cranio-caudal position. In one participant (sub-114), TotalSegmentator produced overlapping L1 and L2 masks that spanned multiple vertebral bodies in both sessions. This was corrected by merging the masks and splitting at the midpoint between the L1 and L2 centroids along the z-axis.

#### Bone Mineral Density

Vertebral bodies were isolated from posterior elements by retaining only the anterior connected component per axial slice and excluding the cranial and caudal 10% (endplates). A 3 mm distance-transform erosion was applied to the isolated vertebral body mask to exclude cortical bone and obtain a trabecular region of interest. Phantom calibration was performed per scan by automatically detecting the cylindrical inserts using Hough circle detection and fitting a linear regression of measured HU values against known densities (R2 range across all 26 scans: 0.998–1.000). Trabecular voxel HU values (filtered to the -50 to 400 HU range) were converted to volumetric bone mineral density (vBMD) in mg/cm3 using the scan-specific calibration equation.

#### Muscle Composition

Left and right autochthonous (erector spinae) muscle masks were combined and a fascial envelope was generated using morphological closing (5 mm spherical structuring element) followed by 2D hole-filling per axial slice, restricted to the L1–L2 z-range. This envelope approach was necessary because TotalSegmentator segments only contractile muscle tissue, excluding intramuscular fat voxels and producing a "Swiss cheese" pattern unsuitable for fat quantification. Within the envelope, voxels were classified by HU after scanner drift correction (estimated from the phantom base insert): muscle tissue (-29 to +150 HU), normal-density muscle (+30 to +150 HU), low-attenuation muscle (-29 to +29 HU), and intermuscular adipose tissue (IMAT; -190 to -30 HU). HU-based IMAT classification was validated against TotalSegmentator's learned intermuscular fat segmentation (tissue_4_types task), with the latter restricted to the same muscle envelope to avoid phantom artifacts.

#### Visceral Adipose Tissue

Visceral adipose tissue (VAT) was quantified using TotalSegmentator anatomical labels restricted to the L1–L2 z-range. However, the spine-focused field of view resulted in adequate abdominal coverage in only 1 of 13 participants; VAT results are therefore not reported at the group level.

### Statistical Analysis

Pre- and post-treatment comparisons were performed using two-tailed paired t-tests. Effect sizes are reported as Cohen's d~z~ (mean change divided by the standard deviation of change) with 95% confidence intervals for the mean difference. All analyses were conducted in Python using SciPy (v1.x). Given the exploratory nature of this pilot study (n = 13), no correction for multiple comparisons was applied.

---

## Results

All 26 scans (13 participants × 2 timepoints) were processed successfully. Phantom calibration R2 exceeded 0.997 in all scans (mean 0.999), and scanner drift was minimal (range: -10 to +4 HU). Bilateral erector spinae left/right symmetry indices ranged from 0.82 to 1.00, indicating consistent muscle segmentation. Mean L1 and L2 trabecular volumes were 7.8 ± 2.6 cm3 and 9.9 ± 3.2 cm3, respectively.

### Bone Mineral Density

Mean L1–L2 trabecular vBMD was 111.9 ± 26.1 mg/cm3 at baseline and 114.1 ± 27.9 mg/cm3 at follow-up (mean change +2.1 ± 7.1 mg/cm3; p = 0.31). L1 and L2 individually showed similar non-significant trends (L1: +2.5 ± 7.4 mg/cm3, p = 0.25; L2: +1.8 ± 7.0 mg/cm3, p = 0.38).

### Muscle Composition

Skeletal muscle density (SMD) was 40.7 ± 8.1 HU at baseline and 40.4 ± 8.6 HU at follow-up (change -0.3 ± 3.5 HU; p = 0.73). Low-attenuation muscle percentage was 30.5 ± 12.3% at baseline and 31.4 ± 12.8% at follow-up (p = 0.54). IMAT percentage was 3.4 ± 2.2% at baseline and 3.7 ± 2.8% at follow-up (p = 0.21). Muscle cross-sectional area showed a trend toward decrease (33.2 ± 9.0 cm2 vs. 32.2 ± 9.7 cm2; change -1.0 ± 1.7 cm2; p = 0.058).

### IMAT Validation

Comparison of HU-based IMAT against TotalSegmentator's learned intermuscular fat segmentation (restricted to the muscle envelope) yielded a mean Dice coefficient of 0.54 (range 0.06–0.75). Dice values exceeding 0.5 were observed in 19 of 26 sessions (73%). Low agreement in the remaining sessions was driven by very small absolute IMAT volumes (<2 cm3), where minor spatial discrepancies disproportionately affect Dice.

### Summary

**Table 1.** Pre–post comparison of CT-derived bone and muscle outcomes.

| Outcome | n | Baseline (SD) | Follow-up (SD) | Δ (95% CI) | % diff | d~z~ | p |
|---------|:-:|---------------|----------------|------------|:------:|:----:|:---:|
| L1–L2 vBMD (mg/cm³) | 13 | 111.9 (26.1) | 114.1 (27.9) | +2.11 (−2.19, +6.41) | +1.9 | +0.30 | .307 |
| L1 vBMD (mg/cm³) | 13 | 110.5 (25.9) | 113.0 (28.2) | +2.46 (−1.98, +6.90) | +2.2 | +0.33 | .251 |
| L2 vBMD (mg/cm³) | 13 | 113.3 (26.7) | 115.1 (27.8) | +1.76 (−2.47, +5.98) | +1.5 | +0.25 | .383 |
| Muscle SMD (HU) | 13 | 40.7 (8.1) | 40.4 (8.6) | −0.34 (−2.44, +1.75) | −0.8 | −0.10 | .729 |
| Low-density muscle (%) | 13 | 30.5 (12.3) | 31.4 (12.8) | +0.90 (−2.20, +4.00) | +3.0 | +0.18 | .539 |
| IMAT (%) | 13 | 3.4 (2.2) | 3.7 (2.8) | +0.33 (−0.21, +0.86) | +9.6 | +0.37 | .209 |
| Muscle volume (cm³) | 13 | 206.6 (55.7) | 201.5 (60.6) | −5.06 (−11.12, +1.00) | −2.5 | −0.50 | .094 |
| Muscle CSA (cm²) | 13 | 33.2 (9.0) | 32.2 (9.7) | −0.97 (−1.98, +0.04) | −2.9 | −0.58 | .058 |

Values are mean (SD). Δ = mean change (follow-up minus baseline) with 95% confidence interval. % diff = mean change relative to baseline. d~z~ = Cohen's d~z~ (mean change / SD of change). p from two-tailed paired t-tests, uncorrected for multiple comparisons.

No statistically significant pre–post differences were observed for any bone or muscle outcome. Muscle CSA showed a trend toward decrease (d~z~ = −0.58, p = .058). The study was not powered to detect small treatment effects (n = 13).
