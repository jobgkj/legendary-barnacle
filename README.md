# Automated XCT Data Analysis for Defect Detection and Segmentation in Additively Manufactured Metal Parts

## 🔬 Project Overview
This project is part of the **PODFAM research initiative** at **University West**. The goal is to develop a robust, automated backend for the segmentation and classification of internal defects (pores, cracks, and inclusions) in X-ray Computed Tomography (XCT) data of Additively Manufactured (AM) metal components.

By transitioning from manual analysis to Deep Learning-based segmentation, this tool aims to improve reliability, standardize defect characterization, and reduce computational overhead in the Quality Assurance (QA) workflow.

## 🚀 Key Features
- **3D Voxel Processing:** Native 3D segmentation using volumetric patches (64x64x64).
- **Advanced Architectures:** Comparative implementation of **3D U-Net** and **Attention U-Net**.
- **Automated Denoising:** Integrated preprocessing pipeline using Non-Local Means (NLM) filtering.
- **Quantitative Metrics:** Automated calculation of **Dice Coefficients** and **Intersection over Union (IoU)** for defect validation.

---

## 🏗️ Architecture Description

### 1. 3D U-Net (Baseline)
The baseline model follows the classic encoder-decoder structure. It utilizes skip connections to concatenate high-resolution features from the contracting path with the upsampled outputs, ensuring that fine-grained spatial details of small defects are preserved.

### 2. Attention 3D U-Net
To address the challenge of segmenting micro-pores and fine cracks amidst imaging artifacts, we implement **Attention Gates (AGs)**. These gates automatically learn to focus on target structures of varying shapes and sizes. By suppressing feature responses in irrelevant background regions, the Attention U-Net achieves higher sensitivity in low-contrast XCT volumes.

---

## 📁 Repository Structure

```text
├── models/
│   ├── unet3d.py           # Standard 3D U-Net implementation
│   └── attention_unet.py   # Attention U-Net with Gating modules
├── preprocessing/
│   ├── denoising.py        # NLM and Median filtering scripts
│   └── patch_extractor.py  # Sliding window 3D patch generation
├── notebooks/
│   ├── training.ipynb      # Training loops and loss curve visualization
│   └── inference.ipynb     # Volume reconstruction and overlap averaging
├── evaluation/
│   └── metrics.py          # Dice, IoU, and Precision/Recall calculations
├── README.md
└── requirements.txt        # Environment dependencies
