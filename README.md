# Hybrid MRI Slice Reconstruction using Deep Learning

A hybrid deep learning framework for reconstructing missing MRI slices and generating high-fidelity 3D MRI volumes by combining classical interpolation techniques with convolutional neural networks.

---

## Problem Statement

Fast MRI acquisition often produces large gaps between slices. While individual 2D slices may appear acceptable, stacking them creates low-quality and discontinuous 3D structures, reducing diagnostic accuracy.

This project addresses the challenge of recovering missing MRI information while preserving anatomical continuity and fine structural details.

---

## Proposed Hybrid Framework

The system combines two complementary approaches:

### Classical Branch (B-Spline Interpolation)

* Generates smooth structural estimates
* Preserves anatomical continuity
* Provides numerical stability

### Deep Learning Branch (CNN)

* Learns anatomical patterns from MRI data
* Restores high-frequency textures and edges
* Enhances visual fidelity beyond interpolation methods

### Adaptive Fusion Module

The outputs from both branches are fused using learnable weights:

Output = α × CNN Reconstruction + (1 − α) × B-Spline Reconstruction

This enables dynamic balancing between structural consistency and detail restoration.

---

## Key Features

* Missing MRI slice reconstruction
* B-Spline interpolation pipeline
* CNN-based enhancement network
* Adaptive fusion strategy
* Confidence map generation
* Quantitative quality evaluation
* High-fidelity 3D MRI reconstruction

---

## Dataset

* BraTS MRI Dataset
* Multi-slice brain MRI scans
* Used for learning anatomical structures and reconstruction patterns

---

## Reconstruction Pipeline

1. MRI Data Acquisition
2. Gap Identification
3. B-Spline Interpolation
4. CNN-Based Enhancement
5. Adaptive Fusion
6. Confidence Map Generation
7. Final High-Resolution 3D MRI Volume

---

## Experimental Results

| Metric | Value   |
| ------ | ------- |
| PSNR   | 36.4 dB |
| SSIM   | 0.98    |

Results demonstrate significant improvements over traditional interpolation approaches while maintaining structural consistency.

---

## Project Structure

```text
Hybrid-MRI-Reconstruction/
│
├── api_server.py
├── demo.py
├── step1_load_data.py
├── step1_verify_processed.py
├── step2_missing_detection.py
├── step3_reconstruction.py
├── step4_weight.py
├── step5_A_train_model.py
├── step5_ml_model.py
├── step6_fusion.py
├── step7_evaluation.py
├── step8_confidence_map.py
├── ml_model.pth
└── README.md
```

---

## Confidence Map Generation

The framework generates voxel-level confidence scores to quantify reconstruction reliability.

* High confidence → reliable reconstruction
* Low confidence → uncertain regions
* Helps identify areas requiring expert review
* Improves trustworthiness of AI-assisted medical imaging systems

---

## Future Work

* Transformer-based MRI reconstruction
* GAN-assisted enhancement
* Full 3D CNN architectures
* Clinical validation studies
* Real-time deployment pipeline

---

## Technologies Used

Python • PyTorch • OpenCV • NumPy • Scikit-Learn • Medical Imaging • Deep Learning • Computer Vision

---

## Author

Harsh Tomar

M.Tech (Computer Science & Engineering)

Indian Institute of Technology (IIT) Mandi
<img width="1484" height="593" alt="WhatsApp Image 2026-04-29 at 1 10 34 AM" src="https://github.com/user-attachments/assets/33772f2b-0b28-41d8-817e-db0525e0b974" />
<img width="1600" height="646" alt="WhatsApp Image 2026-04-29 at 9 19 03 AM" src="https://github.com/user-attachments/assets/d33ee1be-53e7-436f-a7f5-af769186c1e8" />
<img width="1600" height="633" alt="WhatsApp Image 2026-04-29 at 9 19 19 AM" src="https://github.com/user-attachments/assets/aa351cb5-416e-40ba-86b4-4af57ff40e36" />

