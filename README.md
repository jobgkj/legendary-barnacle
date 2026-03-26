# Automated XCT Defect Segmentation and Evaluation for Additively Manufactured Metal Parts

## 🔬 Project Overview
This repository provides an **evaluation and visualization pipeline** for deep learning–based
segmentation of internal defects (pores, cracks, inclusions) in **industrial X-ray Computed
Tomography (XCT)** data of additively manufactured (AM) metal components.

The work is conducted within the **PODFAM research initiative at University West**, and focuses on
**voxel-level defect validation**, enabling reliable and standardized quality assessment of XCT volumes.

This repository represents a **foundational backend component** for XCT defect analysis and is intended
to support ongoing model development and benchmarking efforts.

---

## 🚀 Key Features

- **3D Voxel-Based Processing**  
  Native volumetric processing of industrial XCT data represented as 3D TIFF stacks.

- **3D U-Net Inference**  
  Encoder–decoder architecture optimized for volumetric defect segmentation.

- **Robust Industrial XCT Normalization**  
  Percentile-based intensity normalization to mitigate metal artifacts and beam hardening effects.

- **Quantitative Evaluation Metrics**  
  Automated voxel-wise computation of:
  - Dice Similarity Coefficient (DSC)
  - Intersection over Union (IoU)

- **Interactive 3D Visualization**  
  Single-window 3D surface rendering of:
  - Ground truth defects
  - Predicted defects
  - True-positive overlap regions

---

## 🏗️ Architecture Description

### 3D U-Net (Baseline)
The implemented 3D U-Net follows a classical encoder–decoder structure with skip connections that
preserve spatial context across scales. This architecture is well-suited for industrial XCT data,
where defect morphology varies significantly in size and shape.

The model operates directly on voxel volumes reconstructed from sequential TIFF slices.

---

## 📁 Repository Structure
```bash
industrial-ct-unet-3d/
├── industrial_ct_3d_unet_eval_and_visualize.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── README.md
│   ├── volume/    # Placeholder for XCT slices (not included)
│   └── label/     # Placeholder for ground truth slices (not included)
│
└── results/       # Optional output directory
```
**Note:** Industrial XCT data and trained model weights are intentionally excluded from version control.

---

## 🛠️ Installation & Environment

We recommend using Conda for managing the CUDA-enabled environment.

```bash
conda create -n podfam_env python=3.10
conda activate podfam_env
pip install -r requirements.txt
```
---
▶️ Running the Evaluation
Ensure the dataset is structured as follows (locally):
```bash
dataset/
├── volume/
│   ├── slice_000.tif
│   ├── slice_001.tif
│   └── ...
├── label/
│   ├── slice_000.tif
│   ├── slice_001.tif
│   └── ...
└── model.pth
```
Then run:

```bash
python industrial_ct_3d_unet_eval_and_visualize.py
```
---

📊 Evaluation Output
The script reports:

Dice Similarity Coefficient (DSC)
Intersection over Union (IoU)

An interactive 3D visualization window is launched to support qualitative inspection of the segmentation results.

---
## 🎓 Acknowledgments & References

This research is conducted within the PODFAM project framework at University West.

<a id="1">[1]</a> 
Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas.Ronneberger, O., et al. (2015)

<a id="2">[2]</a> 
Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

<a id="3">[3]</a> 
University West / PODFAM Team for providing XCT datasets and research guidance.

---
Author: Job George Konnoth Joseph

Contact: job-george.konnoth-joseph@student.hv.se
