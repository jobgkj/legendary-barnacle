# Automated XCT Defect Segmentation and Evaluation
## for Additively Manufactured Metal Parts

---

## 🔬 Project Overview

This repository provides a **training, evaluation, and visualization framework** for deep learning–based
segmentation of internal defects (pores, lack‑of‑fusion defects, cracks) in **industrial X‑ray Computed
Tomography (XCT)** data of **additively manufactured (AM) metal components**.

The work is conducted within the **PODFAM research initiative at University West** and focuses on
**voxel‑level defect segmentation and validation**, enabling reliable and reproducible quality
assessment of XCT volumes in metal additive manufacturing.

The repository supports:
- **Model training (2D and 3D U‑Net)**
- **Quantitative benchmarking**
- **Volumetric visualization**

and is intended to serve as a **research backend for XCT defect analysis and method comparison**.

---

## 🚀 Key Features

- **Volumetric XCT Processing**  
  Native handling of industrial XCT data represented as 3D TIFF image stacks.

- **2D & 3D U‑Net Architectures**
  - **2D U‑Net**: slice‑wise baseline for reviewer comparison
  - **3D U‑Net**: patch‑based volumetric model exploiting full 3D context

- **Industrial XCT‑Specific Normalization**  
  Percentile‑based intensity normalization to mitigate:
  - Beam hardening
  - Metal artifacts
  - Scan‑to‑scan intensity variation

- **Training with Class‑Imbalance‑Aware Loss**  
  Combined **Dice + Binary Cross‑Entropy (BCE)** loss to improve sensitivity to small defects.

- **Quantitative Evaluation Metrics**  
  Automated voxel‑wise computation of:
  - Dice Similarity Coefficient (DSC)
  - Intersection over Union (IoU)

- **Interactive 3D Visualization**  
  Volumetric surface rendering of:
  - Ground‑truth defects
  - Predicted defects
  - True‑positive overlap regions

---

## 🏗️ Architecture Description

### 2D U‑Net (Baseline)
A standard encoder–decoder U‑Net applied **slice‑by‑slice** to XCT volumes.  
This baseline evaluates whether in‑plane context alone is sufficient for defect segmentation and serves
as a **mandatory comparison baseline** in AM‑XCT literature.

### 3D U‑Net (Primary Model)
A volumetric U‑Net trained and inferred using **overlapping 3D patches** extracted from XCT volumes.
Patch‑based processing enables robust learning of:
- Volumetric defect connectivity
- Elongated lack‑of‑fusion defects
- Irregular pore morphology

Both models use **Instance Normalization**, which is better suited than Batch Normalization for small
batch sizes typical of XCT workflows.

---

## 🧠 Training Strategy

### Data Sources
Due to the limited availability of fully annotated public AM‑XCT datasets, training data are sourced from:

- **NIST Additive Manufacturing Metrology Testbed (AMMT) XCT datasets**
  (e.g., Overhang Part X4 / X16)
- **NIST high‑resolution LPBF XCT datasets with segmented volumes**
- **In‑house Ti‑6Al‑4V XCT data (PODFAM project)**

Voxel‑level defect labels are generated using **adaptive thresholding followed by expert correction**,
which is consistent with standard practice in AM‑XCT studies.

### Training Setup
- **2D U‑Net**: slice‑wise training
- **3D U‑Net**: patch‑based volumetric training
- **Loss**: Dice + Binary Cross‑Entropy
- **Optimization**: Adam
- **Evaluation**: Held‑out XCT volumes

---

## 📁 Repository Structure

```bash
xct_defect_detection/
│
├── config.py                  # All configuration in one place
├── data/
│   ├── loader.py              # TIFF stack loading
│   ├── pseudo_labels.py       # Otsu pseudo-label generation
│   ├── dataset.py             # PyTorch Dataset + patch extraction
│   └── augmentation.py        # All augmentation transforms
├── models/
│   ├── unet2d.py              # 2D U-Net architecture
│   └── unet3d.py              # 3D U-Net architecture
├── training/
│   ├── losses.py              # BCE, Dice, Focal, Combined losses
│   ├── metrics.py             # Dice, IoU, Precision, Recall
│   └── trainer.py             # Training loop + MLflow logging
├── pipeline.py                # End-to-end pipeline (main entry point)
└── requirements.txt
```

## 🛠️ Installation & Environment

A CUDA‑enabled Python environment is recommended.

```bash
conda create -n podfam_env python=3.10
conda activate podfam_env
pip install -r requirements.txt
```
---
🧪 Training the Models
Prepare local data:
```bash
dataset/
dataset/
├── volume/
│   ├── slice_000.tif
│   ├── slice_001.tif
│   └── ...
├── label/
│   ├── slice_000.tif
│   ├── slice_001.tif
│   └── ...
``
```
Run training:

```bash
python industrial_ct_unet_training.py
```
This produces:

dataset/model_2d.pth
dataset/model_3d.pth
---

▶️ Evaluation & Visualization

```bash
python industrial_ct_unet_2d_vs_3d_eval.py
```
Outputs

Dice Similarity Coefficient (DSC)
Intersection over Union (IoU)
Direct 2D vs 3D comparison
Interactive 3D visualization

---
📊 Intended Use

This repository is designed for:

Research benchmarking (2D vs 3D XCT segmentation)
Development of new architectures (e.g., attention U‑Net, 2.5D models)
Supporting PODFAM and related AM‑XCT research

It is not intended as a production inspection system, but as a research and validation platform.

---
## 🎓 Acknowledgments & References

This work is conducted within the PODFAM research initiative at University West, Sweden.

<a id="1">[1]</a>
Ronneberger, O., Fischer, P., & Brox, T. (2015). 
U‑Net: Convolutional Networks for Biomedical Image Segmentation. 
Proceedings of MICCAI. https://doi.org/10.1007/978-3-319-24574-4_28

<a id="2">[2]</a> 
Oktay, O., Schlemper, J., Folgoc, L. L., et al. (2018). 
Attention U‑Net: Learning Where to Look for the Pancreas. 
arXiv preprint. https://arxiv.org/abs/1804.03999

<a id="3">[3]</a> 
Bellens, S., Vandewalle, P., & Dewulf, W. (2021). 
Deep Learning–Based Porosity Segmentation in X‑ray CT Measurements of Additive Manufacturing Parts. 
Procedia CIRP, 96, 336–341. https://doi.org/10.1016/j.procir.2021.01.157

<a id="4">[4]</a> 
Xu, C., Wang, F., Wei, G., et al. (2024). 
High‑Performance Deep Learning Segmentation for Non‑Destructive Testing of X‑ray Tomography. 
Journal of Manufacturing Processes, 128, 98–110. https://doi.org/10.1016/j.jmapro.2024.08.031

<a id="5">[5]</a> 
Praniewicz, M., Lane, B., Kim, F., & Saldana, C. (2020). 
X‑ray Computed Tomography Data of Additive Manufacturing Metrology Testbed (AMMT) Parts: Overhang Part X4. 
Journal of Research of NIST, 125. https://doi.org/10.6028/jres.125.031

---
Author: Job George Konnoth Joseph

Contact: job-george.konnoth-joseph@student.hv.se
