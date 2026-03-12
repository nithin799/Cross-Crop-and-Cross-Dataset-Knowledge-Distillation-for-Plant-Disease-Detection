# Dataset Setup

This directory should contain the datasets used in this project. They are excluded from version control due to their large size.

## Lentil Augmented Dataset

Place the dataset with this structure:

```
Lentil_Augmented_Dataset/
├── Train/
│   ├── Ascochyta blight/
│   ├── Lentil Rust/
│   ├── Normal/
│   └── Powdery Mildew/
├── Validation/
│   └── (same 4 classes)
└── Test/
    └── (same 4 classes)
```

- Total: 4,540 training / 456 validation / 457 test images

## Beans Dataset

Place the dataset with this structure:

```
Beans_dataset/
├── anthra/      (13,531 images)
├── healthy/     (24,970 images)
└── rust/        (20,568 images)
```

- Total: 59,069 images
- The notebooks split this programmatically into 70% train / 15% validation / 15% test
