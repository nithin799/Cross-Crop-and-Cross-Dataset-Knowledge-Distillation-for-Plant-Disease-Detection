# Saved Models

Model weight files (`.keras`, `.h5`) are excluded from version control due to their large size (~917 MB total).

## Key Model Files

| File | Description | Size |
|------|-------------|------|
| `ResNet101_lentil_10.keras` | **Teacher model** — ResNet101 fine-tuned on Lentil (10 layers unfrozen) | ~211 MB |
| `ResNet101_lentil_05.keras` | ResNet101 on Lentil (5 layers unfrozen) | ~185 MB |
| `ResNet101_10_Beans.keras` | ResNet101 trained on Beans | ~245 MB |
| `Resnet101_10_beans.keras` | ResNet101 baseline on Beans (Without KD) | ~165 MB |
| `Xception_beans.keras` | Xception baseline on Beans | ~135 MB |
| `MobileNet_FD_Best.weights.h5` | MobileNet feature distillation best weights | ~40 MB |
| `EfficientNetB1_beans.keras` | EfficientNetB1 baseline on Beans | ~27 MB |
| `best_student_kd.h5` | Best student from KD training | ~14 MB |
| `Mobilenet_10_beans.keras` | MobileNet baseline on Beans | ~1.6 MB |

## Reproducing Models

To reproduce these models, run the notebooks in order:

1. **Phase 1**: Run notebooks in `notebooks/01_teacher_training_lentil/` to train and select the teacher model
2. **Phase 2**: Run notebooks in `notebooks/02_baseline_beans/` for baseline student models
3. **Phase 3**: Run notebooks in `notebooks/03_knowledge_distillation/` for distilled student models (requires the teacher model from Phase 1)
