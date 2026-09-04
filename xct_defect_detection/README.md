# XCT Defect Segmentation and Evaluation
## for Additively Manufactured Metal Parts

---

## Project Overview

This repository implements a **reproducible, physics-informed workflow** for defect segmentation and evaluation in **industrial X-ray Computed Tomography (XCT)** data of additively manufactured metal components.

The pipeline combines **classical image processing-based segmentation** (global and local thresholding) with **deep learning-based defect segmentation** (a 2.5D U-Net trained on the classical output as pseudo-labels), enabling:

- Objective comparison of **global vs. local thresholding methods** (Otsu, Yen, Bernsen)
- Generation of **weak pseudo-labels** for supervised learning, with no manual ground truth required
- External validation against independently published porosity data
- **Shape-agnostic sample-boundary detection**, matched per sample family — a classical circle-fit method for circular reference samples, and a sliding-window boundary scan for non-circular industrial parts (hexagonal, crescent, trapezoidal cross-sections) where a circular-boundary assumption breaks down
- Quantitative porosity and pore morphology analysis, plus 3D defect visualization

Raw XCT data, trained checkpoints, and the MLflow tracking store are **never committed** to the repository — see `.gitignore`.

---

## Repository Structure

```
podfam_research_project_XCT_Anlaysis/
└── xct_defect_detection/
    ├── config.py                        ← All hyperparameters and paths (env-var overridable)
    ├── pipeline.py                      ← End-to-end training entry point
    ├── gui_app.py                       ← Interactive 3D defect viewer
    ├── launcher_gui.py                  ← Pipeline control panel
    │
    ├── src/
    │   ├── preprocess.py                ← Three-stage production preprocessing pipeline
    │   ├── thresholding.py              ← Bernsen, Otsu, and Yen thresholding
    │   ├── sample_mask.py               ← Sample-boundary detection (circle-fit, peak-scan, Canny)
    │   ├── io.py                        ← Mask generation orchestration
    │   └── metrics.py                   ← Per-pore connected-component statistics
    │
    ├── data/
    │   ├── cache.py                     ← Memory-mapped volume/mask cache
    │   ├── dataset.py                   ← PyTorch Dataset, 2.5D patch extraction
    │   ├── dataset_3d.py                ← 3D patch extraction (3D U-Net, unused by
    │   │                                    default — see config.SKIP_3D_TRAINING)
    │   ├── augmentation.py              ← Six augmentation transforms
    │   ├── raw/                         ← Raw TIFF stacks (not committed)
    │   ├── processed/                   ← Preprocessed 8-bit slices (not committed)
    │   └── masks/                       ← Pseudo-labels for training (not committed)
    │
    ├── models/
    │   ├── unet2d.py                    ← 2.5D U-Net (2D convolutions, multi-slice input)
    │   ├── unet3d.py                    ← 3D U-Net (implemented, disabled by default)
    │   └── thesis_analysis.py           ← Thesis figure-generation script
    │
    ├── training/
    │   ├── losses.py                    ← BCE, Dice, Focal, Dice-Focal
    │   ├── metrics.py                   ← Dice, IoU, Precision, Recall
    │   └── trainer.py                   ← Training loop with MLflow tracking
    │
    ├── scripts/
    │   ├── run_preprocess.py, run_preprocess_bhc_ring.py,
    │   │   generate_training_data.py    ← Preprocessing / mask-generation drivers
    │   ├── run_masks_shape_aware.py,
    │   │   run_masks_no_circle.py       ← Shape-agnostic boundary-mask production runs
    │   ├── regenerate_bernsen_*.py      ← Boundary-method-specific Bernsen mask regeneration
    │   │                                    (material_value / filtered / circlefit / canny)
    │   ├── visualize_3d_bernsen.py,
    │   │   visualize_material_*.py      ← Boundary and 3D-render diagnostics
    │   ├── predict_and_visualize_3d.py  ← Inference and 3D reconstruction
    │   ├── eval_baseline_test_set.py    ← Held-out cross-sample generalisation check
    │   └── train_2d_single_sample.py    ← Plain-2D single-sample training pilot
    │
    ├── data/raw/, data/processed/, data/masks/   ← Input/intermediate data (not in Git)
    ├── mlruns/                          ← MLflow tracking store (not in Git)
    ├── artifacts/                       ← Trained model checkpoints (not in Git)
    └── results/                         ← Figures, predictions, comparison tables
```

---

## STEP 1 — RAW XCT INPUT

Input XCT data are provided as ordered TIFF slice stacks:

```
data/raw/sample_01/
    Sample0001.tiff
    Sample0002.tiff
    ...
```

Slices **must be named such that alphabetical ordering corresponds to the physical build (Z) direction**.

---

## STEP 2 — XCT PREPROCESSING

Two preprocessing entry points are available:

```bash
python scripts/run_preprocess.py            # baseline 3-stage pipeline
python scripts/run_preprocess_bhc_ring.py    # + beam-hardening & ring-artefact correction
```

### Preprocessing Operations

1. **Beam-hardening correction** (BHC+ring pipeline only) — degree-3 polynomial fit to depth-wise mean intensity, removing the artificial intensity drift caused by preferential absorption of lower-energy X-rays with depth.
2. **Ring-artefact suppression** (BHC+ring pipeline only) — polar-coordinate transform + radial median filter, removing detector-calibration ring artefacts.
3. **3D Median Filtering (3×3×3)** — suppresses speckle noise while preserving sharp defect boundaries.
4. **2D Non-Local Means (NLM) Filtering** — noise reduction with edge preservation; noise level estimated automatically from the data.

### Output

```
data/processed/sample_01/               # or data/processed_bhc_ring/ for the corrected pipeline
    Sample0001.tiff
    Sample0002.tiff
    ...
```

> **Note:** the processed directory is the **single shared input source** for all subsequent thresholding, comparison, and learning stages. Run this step before any other. `config.py` exposes `XCT_PROCESSED_DIR` / `XCT_MASKS_DIR` / `XCT_CACHE_DIR` environment-variable overrides for running isolated experiments against an alternate preprocessed tree without disturbing the defaults.

---

## STEP 3 — SAMPLE-BOUNDARY DETECTION

Before thresholding, each slice needs a sample-boundary mask separating the part from background — otherwise dark background air gets misread as porosity. `src/sample_mask.py` implements three methods, matched to sample geometry:

### 1. Circle-fit (`detect_sample_mask`)
Fits an idealised circle from the largest connected component's centroid and equivalent radius. Correct for genuinely circular samples; supports both a fixed and an adaptive (per-slice Otsu) threshold — adaptive is the safer default for samples with a non-uniform background brightness, fixed is needed instead on samples with extreme porosity (>50%) where Otsu's two-class assumption breaks down.

### 2. Sliding-window peak-scan (`detect_sample_mask_peak_scan`)
For non-circular cross-sections (hexagonal, crescent, trapezoidal), where an idealised circle either clips real material at the corners or extends into background. Slides a window across the slice, takes the local histogram peak nearest maximum intensity at each position, and Otsu-splits the resulting coarse map. Requires a larger-than-default erosion radius to remove ~10px of boundary positional slop that Bernsen otherwise misreads as a false defect rim (see `scripts/regenerate_bernsen_material_value.py` for the calibrated production settings).

### 3. Canny zone-detection (`detect_sample_mask_canny_zones`)
An edge-detection-based alternative, validated as a cross-check on circular samples but **not production-safe on non-circular industrial parts** — it can confidently trace surrounding support-scaffold structure instead of the real material on slices where material contrast fades near a build transition. Kept in the codebase as a documented research artefact, not used in the production pipeline for those samples.

---

## STEP 4 — THRESHOLDING METHOD COMPARISON

Three binary segmentation approaches are applied to the preprocessed, boundary-masked data:

### Global Threshold — Otsu
Histogram-based global threshold; often misses low-contrast pores and is biased when one class (solid vs. pore) vastly outnumbers the other.

### Global Threshold — Yen
Entropy-based threshold; tends to over-segment and introduce edge artefacts.

### Local Threshold — Bernsen (primary pseudo-label source)
Adaptive local threshold using sliding windows — `T = (I_max + I_min) / 2`, with the dynamic contrast threshold `DCT` auto-computed per image (18× the local standard deviation of the solid phase, following Kim et al. 2017). Chosen as the primary method because it tolerates the beam-hardening intensity gradients that break global thresholds. Externally validated (Pearson r = 0.995 against independently published NIST CoCr porosity).

Run comparison / regeneration:

```bash
python scripts/regenerate_bernsen_material_value.py sample_06 sample_07   # non-circular parts
python scripts/regenerate_bernsen_circlefit.py sample_01 sample_02 ...    # circular reference samples
```

---

## STEP 5 — PSEUDO-LABEL GENERATION (WEAK SUPERVISION)

Bernsen's local thresholding results are used directly as pseudo-labels for training — no manual ground truth is used at this stage. Masks are cached per sample:

```
data/masks/sample_01/bernsen/
    Sample0001.tiff
    ...
```

Pseudo-labels are generated automatically during `pipeline.py` if not already cached, via `data/cache.py`.

---

## STEP 6 — MODEL TRAINING

```bash
python pipeline.py
```

This script:
1. Loads and preprocesses volumes according to `config.py`'s sample lists
2. Generates or loads cached Bernsen pseudo-labels
3. Splits data into train / val / test (`VAL_SPLIT`, `TEST_SPLIT`)
4. Trains a **2.5D U-Net** (3D U-Net is implemented but disabled by default — `SKIP_3D_TRAINING = True`)
5. Compares four loss functions — BCE, Dice, Focal, Dice-Focal (`training/losses.py`) — tracked via MLflow (`training/trainer.py`)
6. Saves the best checkpoint per run to `artifacts/`

### Key config.py parameters

| Parameter | Description |
|---|---|
| `TRAINING_SAMPLES_OVERRIDE` | Explicit sample list, pinned for controlled ablations |
| `MAX_TRAINING_SAMPLES` | Auto-rank most-defect-rich samples when no override is set |
| `UNET_INPUT_SLICES` | Number of stacked adjacent slices for 2.5D input (default 3) |
| `ENCODER_CHANNELS` | U-Net encoder channel depths, e.g. `[64, 128, 256, 512]` |
| `DROPOUT_RATE` | Bottleneck dropout |
| `BATCH_SIZE_2D` / `BATCH_SIZE_3D` | Batch size per training stage |
| `LEARNING_RATE` | Optimiser learning rate |
| `NUM_EPOCHS` | Max epoch budget |
| `EARLY_STOP_PATIENCE` | Epochs without validation improvement before stopping |
| `VAL_SPLIT` / `TEST_SPLIT` | Data split fractions |
| `LOSS_FUNCTION` | `"bce"`, `"dice"`, `"focal"`, or `"dice_focal"` — env-override via `XCT_LOSS_FUNCTION` |
| `BERNSEN_RADIUS` | Local window radius for Bernsen thresholding |
| `BERNSEN_DCT_AUTO` | Auto-compute DCT per image (recommended) vs. fixed `BERNSEN_DCT` |
| `BERNSEN_LOW_CONTRAST_ADAPTIVE` | Adaptive (solid-peak-anchored) vs. fixed low-contrast fallback threshold |
| `SAMPLE_MASK_EROSION_RADIUS` | Boundary erosion — sample-geometry-dependent, see Step 3 |
| `PIXEL_SIZE_UM` | µm per pixel — update per scanner; not yet calibrated for all sample families, see Known Limitations |
| `DEVICE` | `cuda` or `cpu` |

---

## STEP 7 — EVALUATION AND VISUALIZATION

```bash
python scripts/predict_and_visualize_3d.py sample_06
python scripts/eval_baseline_test_set.py
```

### Quantitative Metrics
- Dice Similarity Coefficient, IoU, Precision, Recall (validation and genuinely held-out cross-sample test)
- Correlation (Pearson r) against independently published reference porosity
- Porosity and pore size distributions (`src/metrics.py`)

### Qualitative Analysis
- Grayscale + mask overlays, slice-by-slice comparison figures
- 3D defect reconstruction (marching-cubes mesh, orthographic MIP, defect scatter)
- `gui_app.py` — interactive 3D viewer

---

## Known Limitations

- The core validation set's held-out cross-sample test reveals a real generalisation gap: validation performance does not predict performance on a genuinely unseen sample (see thesis Results/Discussion) — training on only a handful of samples is the primary driver, not any single preprocessing choice.
- Training labels are pseudo-labels derived from Bernsen thresholding the same XCT reconstructions used for evaluation — there is no independent, manually-annotated ground truth at this stage.
- `PIXEL_SIZE_UM` was not calibrated to every sample family's actual scan resolution; some figures are reported in pixels rather than physical units as a result — check raw TIFF metadata (`XResolution`/`YResolution` tags) per sample before assuming a shared scale.
- `pipeline.py` loads full volumes into RAM — large datasets may require significant memory.
- `src/io.py` and downstream stages read from the processed directory — ensure Step 2 has been run first.

---

## Notes on Reproducibility

- Threshold parameters are derived from the data itself wherever possible (Bernsen's DCT, Otsu, robust stack-wide solid-level references) rather than reused as fixed constants across datasets.
- Raw XCT data, trained checkpoints, and the MLflow store are excluded from version control via `.gitignore`.
- `config.py` supports environment-variable overrides (`XCT_PROCESSED_DIR`, `XCT_MASKS_DIR`, `XCT_CACHE_DIR`, `XCT_LOSS_FUNCTION`, `XCT_EXPERIMENT_NAME`, `XCT_UNET_INPUT_SLICES`) so experiments can be run against isolated data/cache directories without mutating the defaults.

---

**Author:** Job George Konnoth Joseph
**Contact:** job-george.konnoth-joseph@student.hv.se
