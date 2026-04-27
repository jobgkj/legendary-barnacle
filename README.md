# XCT Defect Segmentation and Evaluation

## for Additively Manufactured Metal Parts

---

##  Project Overview

This document describes the complete processing, training, and evaluation
workflow implemented in this repository for automated defect segmentation
in industrial X-ray Computed Tomography (XCT) data of additively manufactured
metal components.

The same data, preprocessing steps, and pseudo-labels are used to train
both 2D and 3D U-Net models, enabling a fair and controlled comparison
between slice-wise and volumetric learning.

---------------------------------------------------------------------
STEP 1 — RAW XCT INPUT
---------------------------------------------------------------------

Raw XCT data are provided as ordered TIFF slice stacks:

data/tiff_stack/
    slice_0000.tif
    slice_0001.tif
    slice_0002.tif
    ...

Slices must be named such that alphabetical ordering corresponds to
the physical Z-order of the volume.

---------------------------------------------------------------------
STEP 2 — XCT PREPROCESSING
---------------------------------------------------------------------

The raw TIFF stack is preprocessed using an XCT-specific pipeline
designed for metal additive manufacturing data.

Run:
    python run_preprocess.py

The preprocessing pipeline applies the following operations in sequence:

1. Percentile-based intensity normalization
   - Robust to metal artefacts and extreme outliers

2. Beam hardening correction
   - Polynomial correction of systematic intensity gradients

3. Ring artefact suppression
   - Radial profile correction per slice

4. Non-local means denoising
   - Noise suppression while preserving defect boundaries

The output is a float32 volume normalized to the range [0, 1], saved
slice-wise:

data/tiff_output/
    slice_0000.tif
    slice_0001.tif
    ...

These preprocessed slices form the single shared input source for all
subsequent stages.

---------------------------------------------------------------------
STEP 3 — PSEUDO-LABEL GENERATION (WEAK SUPERVISION)
---------------------------------------------------------------------

Since voxel-accurate ground truth annotations are typically unavailable,
binary pseudo-label masks are generated automatically.

The pseudo-label generation pipeline consists of:

1. Global Otsu thresholding
   - Defects correspond to low X-ray attenuation regions

2. Morphological cleaning
   - Opening removes isolated noise
   - Closing fills small holes in defect regions

3. Connected-component filtering
   - Removal of components smaller than a minimum voxel count

Pseudo-labels are generated once and cached to disk:

    python data/pseudo_labels.py

Resulting files:

data/masks/
    volume_01_mask.tif
    volume_02_mask.tif
    ...

These masks are weak supervision targets, but provide sufficient
structural guidance to train U-Net models that generalize better
than thresholding alone.

---------------------------------------------------------------------
STEP 4 — DATASET CONSTRUCTION (SHARED FOR 2D AND 3D)
---------------------------------------------------------------------

The same preprocessed volumes and pseudo-label masks are used for both
2D and 3D training. No preprocessing, normalization, or labeling differs
between the models.

Two dataset strategies are implemented:

2D DATASET (SLICE-WISE)
----------------------
- Each XCT slice is treated independently
- 2D patches are extracted using a sliding window
- Foreground/background stratified sampling addresses severe class imbalance
- Online 2D augmentation is applied during training

This dataset is used to train the 2D U-Net baseline model.

3D DATASET (PATCH-WISE)
----------------------
- Random 3D patches (D, H, W) are extracted from the full volume
- The same 2D augmentation is applied slice-wise within each 3D patch
- No augmentation is applied across the Z dimension

This approach preserves volumetric consistency while avoiding
unphysical 3D warping.

Both datasets use identical data and pseudo-labels, enabling a fair
comparison between 2D and 3D learning approaches.

---------------------------------------------------------------------
STEP 5 — MODEL TRAINING
---------------------------------------------------------------------

2D U-NET TRAINING
----------------
- Slice-wise training
- Input shape: (B, 1, H, W)
- Inference is performed slice-by-slice and stacked into a 3D volume

3D U-NET TRAINING
----------------
- Patch-based volumetric training
- Input shape: (B, 1, D, H, W)
- Inference is performed using a sliding-window strategy

Common training configuration:
- Loss function: Dice + Binary Cross-Entropy
- Optimizer: Adam or AdamW
- Batch size: small (typically 1–4 due to GPU memory constraints)

Training commands:
    python pipeline.py

Saved models:
artifacts/
    model_2d.pt
    model_3d.pt

---------------------------------------------------------------------
STEP 6 — EVALUATION AND VISUALIZATION
---------------------------------------------------------------------

Models are evaluated both quantitatively and qualitatively.

Quantitative metrics:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)

Qualitative analysis:
- Slice-wise overlay visualization
- Interactive 3D defect surface rendering

Evaluation command:
    python evaluate_2d_vs_3d.py

This enables direct and reproducible comparison of slice-wise (2D)
and volumetric (3D) defect segmentation performance.

---------------------------------------------------------------------
END OF PROCEDURE
---------------------------------------------------------------------
Author: Job George Konnoth Joseph

Contact: job-george.konnoth-joseph@student.hv.se
