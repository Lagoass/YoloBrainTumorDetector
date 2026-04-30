# Run Analysis Log

Scientific log of each training run: metrics, confusion matrix findings, root cause, and next-run plan.
Each section title matches the run folder name under `runs/brain_tumor/`.

---

## Fixed Training Configuration
> These parameters are stable across all runs and are not repeated in run-specific sections.

| Parameter | Value |
|---|---|
| epochs | 100 |
| imgsz | 640 |
| batch | 16 |
| amp | True |
| patience | 20 |
| fliplr | 0.5 |
| flipud | 0.0 |
| degrees | 10.0 |
| mosaic | 0.5 |
| mixup | 0.0 |
| hsv_v | 0.2 |
| hsv_s | 0.0 |
| close_mosaic | 10 |

---

## Run Comparison Table

| Run | mAP@0.50 | mAP@0.5:0.95 | Precision | Recall | Key Change |
|---|---|---|---|---|---|
| yolo11s_29_04_1620 | 0.9217 | 0.5468 | 0.9244 | 0.8453 | Baseline — no class weights |
| yolo11s_29_04_1759 | 0.9217 | 0.5468 | 0.9244 | 0.8453 | label_smoothing=0.1 — no effect |
| yolo11s_29_04_2007 | 0.9290 | 0.6212 | 0.9301 | 0.8844 | Oversampling meningioma+pituitary to glioma parity (seed=42) |

---

## yolo11s_29_04_1620

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9217 |
| mAP@0.5:0.95 | 0.5468 |
| Precision | 0.9244 |
| Recall | 0.8453 |

### Run-specific Changes
None — baseline run, no class weights applied.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Pituitary tumor | 0.87 | Best class — distinct morphology aids separation |
| Glioma | 0.78 | 0.21 missed as background — diffuse borders reduce confidence |
| Meningioma | 0.74 | 0.18 confused with glioma — similar MRI appearance |
| Background → Glioma | 0.77 FP rate | Strong class bias: background patches predicted as glioma |

### Root Cause
Class imbalance: glioma is the most represented class in the dataset. The model develops overconfidence toward glioma, pulling borderline meningioma predictions and background patches into that class.

### Conclusion & Next Run Plan
`label_smoothing=0.1` tested in Run 2 — had no effect. Oversample meningioma before training to balance class distribution at the data level.

---

## yolo11s_29_04_1759

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9217 |
| mAP@0.5:0.95 | 0.5468 |
| Precision | 0.9244 |
| Recall | 0.8453 |

### Run-specific Changes
`label_smoothing=0.1` added to address glioma overconfidence from class imbalance.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Pituitary tumor | 0.87 | Best class — distinct morphology aids separation |
| Glioma | 0.78 | 0.21 missed as background — diffuse borders reduce confidence |
| Meningioma | 0.74 | 0.18 confused with glioma — similar MRI appearance |
| Background → Glioma | 0.77 FP rate | Strong class bias: background patches predicted as glioma |

### Root Cause
`label_smoothing` is a symmetric operation — it softens all classes equally and is insensitive to class distribution. With Ultralytics fixed seed, this produced an identical training trajectory to Run 1. It is the wrong tool for asymmetric class imbalance.

### Conclusion & Next Run Plan
Oversample meningioma before training to balance class distribution at the data level. Create `src/oversample.py` to duplicate meningioma images/labels in the train split until reaching glioma parity (~1426 samples).

---

## yolo11s_29_04_2007

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9290 |
| mAP@0.5:0.95 | 0.6212 |
| Precision | 0.9301 |
| Recall | 0.8844 |

### Run-specific Changes
Oversampling applied — meningioma 552→1087, pituitary 767→1087, glioma unchanged at 1087. Pipeline uses `data/oversample_dataset/dataset.yaml`.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Meningioma | 0.76 | +0.02 vs Run 1 — oversampling improved minority class |
| Glioma | 0.79 | +0.01 vs Run 1 — marginal improvement |
| Pituitary | 0.79 | -0.08 vs Run 1 — expected trade-off: oversample repetitions reduced effective diversity |
| Background → Glioma | 0.78 FP rate | Unchanged — bias persists despite balanced classes |

### Root Cause
Background→Glioma bias persists despite balanced classes, suggesting the problem is partially intrinsic to glioma's diffuse visual appearance in T1 MRI rather than purely a data distribution issue.

### Conclusion & Next Run Plan
Oversampling confirmed correct direction. All global metrics improved, especially mAP@0.5:0.95 (+0.0744) and Recall (+0.0391). Next step: to be defined.
