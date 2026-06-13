# Cross-Crop and Cross-Dataset Knowledge Distillation for Plant Disease Detection

## Problem Statement

Plant diseases cause significant crop losses worldwide, and early detection through automated systems is critical for food security. While deep learning has shown promising results in plant disease classification, training accurate models requires large, well-labeled datasets for each individual crop species — a resource that is often scarce or expensive to obtain.

**This project addresses the following research question:** *Can diagnostic knowledge learned from one crop species be effectively transferred to a different but botanically related crop species using knowledge distillation, thereby improving classification performance on the target crop without requiring additional labeled data from the source domain?*

Specifically, we investigate whether a ResNet101 model trained to classify diseases in **lentil plants** (Family: *Fabaceae*) can serve as a teacher to improve student models classifying diseases in **bean plants** (also *Fabaceae*). The taxonomical similarity between these crops — shared physiological characteristics, leaf morphology, and overlapping disease manifestations (e.g., rust, blight) — provides a scientific basis for this cross-crop transfer.

## Overview

This research project investigates **cross-crop feature-based knowledge distillation** to transfer learned disease representations from a lentil disease model (teacher) to a bean disease classifier (student). The approach leverages intermediate feature representations rather than just output logits, enabling richer knowledge transfer across datasets with different numbers of classes (4 lentil classes vs. 3 bean classes).

### Key Contributions

- Comprehensive benchmarking of **18 CNN architectures** on lentil disease classification
- Fine-tuning depth ablation study on ResNet101 (5 to 30 unfrozen layers)
- Feature-based knowledge distillation using intermediate hint layers
- Cross-crop transfer from lentil (4-class) to bean (3-class) disease detection
- Comparison of KD-assisted vs. baseline student models across 5 architectures

## Project Structure

```
├── notebooks/
│   ├── 01_teacher_training_lentil/       # 19 notebooks: 18 architectures + finetuning ablation study
│   ├── 02_baseline_beans/                # 5 student models trained on Beans (no KD benchmark)
│   ├── 03_knowledge_distillation/        # Student models trained with cross-crop feature KD
│   │   ├── first layers/                 # KD targeting student's first block of layers
│   │   ├── middle layers/                # KD targeting student's middle block of layers
│   │   └── last layers/                  # KD targeting student's last block of layers
│   └── 04_quantization_aware_training/   # Model quantization (post-training INT8 conversion & comparison)
├── results/
│   ├── 01_teacher_lentil/                # Confusion matrices, ROC curves, loss/accuracy plots for teachers
│   ├── 02_finetuning_ablation/           # ResNet101 fine-tuning layer depth experiments
│   ├── 03_student_without_kd/            # Baseline performance curves for students trained without KD
│   ├── 04_student_model/                 # Classification reports, CM, ROC for layer-wise KD
│   └── 05_quantization/                  # Size, accuracy, and latency metrics before/after quantization
├── docs/
│   ├── literature/                       # Reference and review literature papers
│   ├── figures/                          # System diagrams and dataset samples
│   └── reports/                          # Final report, presentation, and project documents
├── data/                                 # Datasets (not tracked, see setup below)
└── models/                               # Saved model weights (not tracked)
```

## Methodology

### Phase 1: Teacher Model Training (Source Domain — Lentil)

- **Dataset**: Lentil Augmented Dataset — 4,550 images (predefined train/val/test splits).
- **Classes**: Ascochyta Blight, Lentil Rust, Normal, Powdery Mildew.
- **Models Evaluated**: 18 architectures including ResNet (50, 101, 152), DenseNet (169, 201), VGG (16, 19), NASNet (Mobile, Large), Inception (V3, ResNetV2), MobileNet (V1, V2), Xception, and others.
- **Best Teacher**: ResNet101 backbone pretrained on ImageNet.
- **Adaptive Fine-Tuning Study**: Sequence of 5, 10, 15, 20, 25, and 30 unfrozen layers from the back of the network. The **10-layer configuration** was selected as the optimal choice (83% accuracy, 5.52M trainable parameters) balancing parameter efficiency and generalization capability.

### Phase 2: Baseline Student Training (Target Domain — Beans)

- **Dataset**: Beans Dataset — 59,069 images (70/15/15 split: 41,350 train, 8,860 val, 8,860 test).
- **Classes**: Anthracnose, Healthy, Rust.
- **Student Architectures**: ResNet101, Xception, MobileNet, DenseNet121, EfficientNetB1.
- **Training Setup**: Standard supervised training on CPU without distillation assistance. 10 epochs, learning rate 1e-6, batch size 32.

### Phase 3: Cross-Crop Knowledge Distillation

- **Lentil Teacher**: Pre-trained ResNet101 (last 10 layers unfrozen version, frozen during distillation).
- **Students**: Same 5 architectures as Phase 2.
- **Distillation Framework**: Feature-based distillation where the student matches intermediate features of the teacher using 1×1 Conv2D projection layers to align dimensions.
- **Layer-wise Configuration Study**: Investigated knowledge transfer by targeting different layer depths:
  - **First Layers**: Aligning first-block feature maps.
  - **Middle Layers**: Aligning mid-block feature maps.
  - **Last Layers**: Aligning semantic/late-block feature maps.

### Phase 4: Model Quantization

- **Technique**: Post-training integer quantization (INT8) using TensorFlow Model Optimization.
- **Evaluation**: Assessed the impacts of quantization on accuracy, model size, and inference latency (measured as average milliseconds per image on CPU) to examine edge deployment feasibility.

---

## Datasets

Datasets are excluded from the repository due to size constraints.

### Lentil Augmented Dataset

Source: Bangladesh field data (Mahamud et al., 2025). Predefined splits are used:

| Class            | Original Images | Augmented Images |
|------------------|-----------------|------------------|
| Normal           | 454             | 1,321            |
| Ascochyta Blight | 466             | 1,029            |
| Powdery Mildew   | 492             | 1,125            |
| Lentil Rust      | 480             | 1,075            |
| **Total**        | **1,898**       | **4,550**        |

### Beans Dataset

Source: Tanzania field data. Split into Train (41,350), Val (8,860), and Test (8,860):

| Class       | Images     |
|-------------|------------|
| Anthracnose | 13,531     |
| Healthy     | 24,970     |
| Rust        | 20,568     |
| **Total**   | **59,069** |

---

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/nithin799/Cross-Crop-and-Cross-Dataset-Knowledge-Distillation-for-Plant-Disease-Detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your datasets under the `data/` folder following the dataset paths.

## Requirements

- Python 3.10
- TensorFlow >= 2.15.0 (tf-keras >= 2.15.0)
- TensorFlow Model Optimization >= 0.8.0
- NumPy >= 1.24.0
- Scikit-learn >= 1.3.0
- OpenCV >= 4.8.0
- Pillow >= 10.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0

---

## Results

### 1. Teacher Baseline & Pre-trained CNNs on Lentil

Evaluation of pre-trained architectures on the Lentil dataset (prior to adaptive retraining):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
|-------|----------|-----------------|--------------|----------|--------------------|-----------------|-------------|
| **ResNet101** | **83%** | **85%** | **83%** | **81%** | **83%** | **78%** | **79%** |
| DenseNet201 | 79% | 79% | 81% | 79% | 79% | 81% | 79% |
| DenseNet169 | 78% | 81% | 80% | 79% | 81% | 78% | 78% |
| ResNet152 | 78% | 81% | 81% | 80% | 81% | 80% | 80% |
| InceptionV3 | 76% | 77% | 76% | 76% | 77% | 74% | 74% |
| VGG16 | 76% | 76% | 78% | 77% | 77% | 76% | 76% |
| VGG19 | 75% | 75% | 76% | 75% | 74% | 75% | 74% |
| MobileNet | 73% | 76% | 76% | 73% | 77% | 73% | 72% |
| MobileNetV2 | 70% | 71% | 72% | 70% | 71% | 70% | 69% |
| Xception | 67% | 68% | 70% | 67% | 69% | 70% | 67% |
| NASNetLarge | 66% | 74% | 67% | 68% | 73% | 66% | 67% |
| NASNetMobile | 63% | 63% | 65% | 63% | 62% | 63% | 61% |

### 2. ResNet101 Fine-Tuning Ablation Study

Impact of unfreezing different layer depths in the ResNet101 backbone:

| Retrained Layers | Accuracy | Trainable Params (M) | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | 81% | 2.11 | 81% | 83% | 81% | 82% | 81% | 81% |
| **10** | **83%** | **5.52** | **83%** | **85%** | **83%** | **83%** | **83%** | **82%** |
| 15 | 82% | 6.57 | 83% | 84% | 83% | 83% | 82% | 82% |
| 20 | 79% | 9.98 | 80% | 82% | 79% | 81% | 79% | 78% |
| 25 | 84% | 11.04 | 84% | 86% | 85% | 84% | 84% | 84% |
| 30 | 81% | 15.50 | 82% | 83% | 81% | 82% | 81% | 80% |

### 3. Baseline vs. Feature-Based Knowledge Distillation (Beans)

Distillation performance using the last-layer (semantic) distillation configurations compared to baseline students:

| Student Architecture | Accuracy (No KD) | Accuracy (With KD) | Accuracy Gain | FP32 Model Size (MB) |
|---|:---:|:---:|:---:|:---:|
| **ResNet101** | 85.20% | 98.86% | **+13.66%** | 167.95 MB |
| **EfficientNetB1** | 67.80% | 96.38% | **+28.58%** | 39.18 MB |
| **Xception** | 81.20% | 94.64% | **+13.44%** | 98.95 MB |
| **DenseNet121** | 71.50% | 93.70% | **+22.20%** | 40.32 MB |
| **MobileNet** | 74.60% | 92.11% | **+17.51%** | 23.14 MB |

### 4. Layer Configuration KD Study

Comparative results of transferring features from different blocks (First, Middle, or Last layers) to the student models:

| Student Architecture | First Layers KD Accuracy | Middle Layers KD Accuracy | Last Layers KD Accuracy |
|---|:---:|:---:|:---:|
| **ResNet101** | 86.00% | 81.00% | **98.86%** |
| **EfficientNetB1** | 79.00% | 82.00% | **96.38%** |
| **Xception** | 76.00% | 76.00% | **94.64%** |
| **DenseNet121** | 63.00% | 80.00% | **93.70%** |
| **MobileNet** | 67.00% | 76.00% | **92.11%** |

### 5. Post-Training Quantization (INT8) Trade-offs

Evaluating the performance drops, model compression rates, and inference speed improvements (measured on CPU):

| Model | Setup | Test Accuracy | Model Size | Latency | Size Reduction | Latency Gain |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **ResNet101** | FP32 KD <br> INT8 Quantized | 98.86% <br> 97.48% | 167.95 MB <br> 42.99 MB | 164.40 ms <br> 194.68 ms | **74.40%** | +18.42% (Slower) |
| **EfficientNetB1** | FP32 KD <br> INT8 Quantized | 96.38% <br> 82.92% | 39.18 MB <br> 8.16 MB | 162.86 ms <br> 54.98 ms | **79.18%** | **-66.24% (Faster)** |
| **Xception** | FP32 KD <br> INT8 Quantized | 94.64% <br> 65.45% | 98.95 MB <br> 21.13 MB | 76.72 ms <br> 140.86 ms | **78.64%** | +83.60% (Slower) |
| **DenseNet121** | FP32 KD <br> INT8 Quantized | 93.70% <br> 36.81% | 40.32 MB <br> 7.04 MB | 187.94 ms <br> 175.39 ms | **82.53%** | **-6.68% (Faster)** |
| **MobileNet** | FP32 KD <br> INT8 Quantized | 92.11% <br> 90.67% | 23.14 MB <br> 3.49 MB | 36.45 ms <br> 27.77 ms | **84.92%** | **-23.81% (Faster)** |

---

## Documentation

- **Final Report**: [`docs/reports/Report.pdf`](docs/reports/Report.pdf)
- **Presentation**: [`docs/reports/Presentation.pdf`](docs/reports/Presentation.pdf)
- **Detailed Distillation Log**: Available in [`results/04_student_model/Results_KD.docx`](results/04_student_model/Results_KD.docx)
- **Detailed Quantization Log**: Available in [`results/05_quantization/Student_model.docx`](results/05_quantization/Student_model.docx)
- **Reference Literature**: Papers located in [`docs/literature/`](docs/literature/)
