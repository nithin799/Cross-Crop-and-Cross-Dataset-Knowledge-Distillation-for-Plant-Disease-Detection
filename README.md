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
│   ├── 01_teacher_training_lentil/       # 19 notebooks: 18 architectures + finetuning ablation
│   ├── 02_baseline_beans/                # 5 student models trained on Beans (no KD)
│   └── 03_knowledge_distillation/        # 5 student models with cross-crop KD
├── results/
│   ├── teacher_lentil/                   # Confusion matrices, ROC curves, loss/accuracy plots
│   ├── finetuning_ablation/              # ResNet101 fine-tuning layer experiments
│   └── result_tables/                    # Summary result tables (.docx)
├── docs/
│   ├── literature/                       # Reference papers
│   ├── figures/                          # Flowcharts and dataset visualizations
│   └── reports/                          # Final report, presentation, and project reports
├── data/                                 # Datasets (not tracked, see setup below)
└── models/                               # Saved model weights (not tracked)
```

## Methodology

### Phase 1: Teacher Model Training (Source Domain — Lentil)

- **Dataset**: Lentil Augmented Dataset — 4,540 train / 456 validation / 457 test images
- **Classes**: Ascochyta Blight, Lentil Rust, Normal, Powdery Mildew
- **Models Evaluated**: VGG (16, 19), ResNet (50, 50V2, 101, 101V2, 152, 152V2), DenseNet (121, 169, 201), Inception (V3, ResNetV2), MobileNet (V1, V2), NASNet (Mobile, Large), Xception
- **Best Teacher**: ResNet101 — 82% baseline, **83% after fine-tuning** (last 10 layers unfrozen)
- **Ablation**: Fine-tuning depth study with 5, 10, 15, 20, 25, and 30 unfrozen layers

### Phase 2: Baseline Student Training (Target Domain — Beans)

- **Dataset**: Beans Dataset — ~59,069 images (70/15/15 train/val/test split)
- **Classes**: Anthracnose, Healthy, Rust
- **Student Architectures**: ResNet101, MobileNet, DenseNet121, EfficientNetB1, Xception
- **Training**: Standard supervised learning without knowledge distillation

### Phase 3: Cross-Crop Knowledge Distillation

- **Teacher**: ResNet101 trained on Lentil (frozen)
- **Students**: Same 5 architectures as Phase 2
- **KD Method**: Feature-based distillation using intermediate hint layers
  - Teacher hint layers: `conv4_block23_out`, `conv5_block3_out`
  - Student features projected via 1×1 Conv2D to match teacher channel dimensions
  - Loss: `total_loss = classification_loss + 0.2 × normalized_feature_MSE`

## Datasets

Datasets are not included in this repository due to size constraints.

### Lentil Augmented Dataset

| Class            | Train | Validation | Test |
|------------------|-------|------------|------|
| Ascochyta Blight | 1,029 | 111        | 111  |
| Lentil Rust      | 1,075 | 96         | 95   |
| Normal           | 1,321 | 142        | 143  |
| Powdery Mildew   | 1,115 | 107        | 108  |
| **Total**        | **4,540** | **456** | **457** |

### Beans Dataset

| Class        | Images |
|-------------|--------|
| Anthracnose | 13,531 |
| Healthy     | 24,970 |
| Rust        | 20,568 |
| **Total**   | **59,069** |

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download datasets and place them in the `data/` directory (see `data/README.md` for structure)
4. Update dataset paths in notebooks if needed

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- GPU recommended for training

See `requirements.txt` for the full dependency list.

## Results

Detailed results including confusion matrices, ROC curves, and training curves are available in the `results/` directory. Summary result tables are in `results/result_tables/`.

## Documentation

- **Final Report**: [`docs/reports/Report.pdf`](docs/reports/Report.pdf)
- **Presentation**: [`docs/reports/Presentation.pdf`](docs/reports/Presentation.pdf)
- **Result Tables**: Available in [`results/result_tables/`](results/result_tables/)
- **Literature Review**: Reference papers in [`docs/literature/`](docs/literature/)
