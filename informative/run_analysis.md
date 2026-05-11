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
| patience | 30 |
| cos_lr | True |
| fliplr | 0.5 |
| flipud | 0.0 |
| degrees | 10.0 |
| mosaic | 0.5 |
| mixup | 0.0 |
| hsv_v | 0.2 |
| hsv_s | 0.0 |
| close_mosaic | 10 |

---

> **Data integrity note:** Runs 1–3 were trained on corrupted 2.5D data. The Z-slice grouping by PID was incorrect — adjacent slices belonged to different patients, producing random channel stacking with no anatomical meaning. Run 4 onwards uses corrected pure 2D inputs (each .mat converted individually, min-max normalized, replicated to 3-channel RGB). All 3064 images must be regenerated before Run 4.

## Run Comparison Table

| Run | Folder | mAP@0.50 | mAP@0.5:0.95 | Precision | Recall | Key Change |
|---|---|---|---|---|---|---|
| 1 | yolo11s_29_04_1620 | 0.9217 | 0.5468 | 0.9244 | 0.8453 | Baseline — no class weights |
| 2 | yolo11s_29_04_1759 | 0.9217 | 0.5468 | 0.9244 | 0.8453 | label_smoothing=0.1 — no effect |
| 3 | yolo11s_29_04_2007 | 0.9290 | 0.6212 | 0.9301 | 0.8844 | Oversampling meningioma+pituitary to glioma parity (seed=42) |
| 4 | yolo11s_03_05_2240 | 0.9236 | 0.5659 | 0.8763 | 0.8723 | 2D puro (sem 2.5D) + cos_lr=True + patience=15 + sem oversample |
| 5 | yolo11s_04_05_1246 | 0.9337 | 0.5549 | 0.9250 | 0.9054 | 2D puro + oversampling (meningioma 552→1087, pituitary 767→1087) |
| 6 | yolo11s_11_05_1536 | 0.8469 | 0.4796 | 0.7844 | 0.8461 | BRISC 2025 — balanced classes + 967 healthy negatives + 3 planes |
| 7 | yolo11s_11_05_1621 | 0.9195 | 0.5907 | 0.9178 | 0.8840 | BRISC 2025 + patience=30 — full 100 epochs, best epoch 62 |

---

## Run 1 — yolo11s_29_04_1620

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

## Run 2 — yolo11s_29_04_1759

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

## Run 3 — yolo11s_29_04_2007

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
Oversampling confirmed correct direction. All global metrics improved, especially mAP@0.5:0.95 (+0.0744) and Recall (+0.0391).

Post-run analysis: `best.pt` was saved at epoch 36, with severe oscillation for the remaining 64 epochs — a clear sign of LR instability after the warmup phase ends. Fix: `cos_lr=True` for smooth monotonic decay + `patience=15` to stop earlier if no improvement.

---

## Run 4 — yolo11s_03_05_2240

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9236 |
| mAP@0.5:0.95 | 0.5659 |
| Precision | 0.8763 |
| Recall | 0.8723 |

### Run-specific Changes
First run on corrected pure 2D data (min-max normalized, 3-channel RGB replicated). No oversampling. cos_lr=True, patience=15. Early stop at epoch 55.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Meningioma | 0.80 | +0.06 vs Run 1 — largest single-class gain across all runs. Clean 2D data directly improved the hardest class. |
| Glioma | 0.71 | -0.07 vs Run 1 — regression. 2.5D noise accidentally reinforced diffuse glioma textures. Without it, model is less confident on glioma. |
| Pituitary | 0.85 | -0.02 vs Run 1 — stable, minor drop. |
| Background → Glioma | 0.68 FP rate | -0.09 vs Run 1 — significant improvement. Clean inputs reduced false positive glioma predictions on background. |

### Root Cause
Pure 2D inputs improved data quality and reduced background bias. However, class imbalance (glioma 46.5% of dataset) now dominates without the accidental texture reinforcement from corrupted 2.5D channels. Glioma needs oversampling compensation on clean data.

### Conclusion & Next Run Plan
Run 5 = 2D puro + oversampling. Expected to combine meningioma gains from clean data with glioma recovery from balanced classes.

---

## Run 5 — yolo11s_04_05_1246

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9337 |
| mAP@0.5:0.95 | 0.5549 |
| Precision | 0.9250 |
| Recall | 0.9054 |

### Run-specific Changes
2D puro + oversampling (meningioma 552→1087, pituitary 767→1087). Best epoch: 86/100. Full 100 epochs, no early stop. Highest Recall of all runs.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Meningioma | 0.74 | Lost Run 4 gain; oversample duplicates reduced effective diversity |
| Glioma | 0.81 | +0.10 vs Run 4 — recovered strongly; oversampling compensated class imbalance |
| Pituitary | 0.93 | Best result of any class across all runs |
| Background → Glioma | 0.87 FP rate | Regression vs Run 4 (0.68); overconfidence from memorized duplicates |

### Root Cause
Oversample with exact duplicates causes memorization — val-train divergence 0.784 vs 0.497 in Run 4. Model associates uncertainty with glioma since it is the only class with fully unique, diverse images. Meningioma and pituitary oversamples are repeated patterns; glioma is not.

### Conclusion & Next Run Plan
Dataset is the bottleneck. 708 meningioma images are insufficient for robust generalization. Undersampling glioma discards valid data without improving meningioma. Priority: find a larger, balanced dataset with healthy brain scans and multiple acquisition planes. Awaiting Deep Research results.

---

## Run 6 — yolo11s_11_05_1536

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.8469 |
| mAP@0.5:0.95 | 0.4796 |
| Precision | 0.7844 |
| Recall | 0.8461 |

### Run-specific Changes
First run on BRISC 2025 dataset. 4802 train / 599 val / 599 test. Stratified 80/10/10 split by class+plane. 967 healthy brain negative samples included. Early stop at epoch 30/100, best epoch 26.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Meningioma | 0.94 | Best result of any class across all runs. 1635 unique diverse images resolved the generalization bottleneck. |
| Glioma | 0.76 | Solid, stable result |
| Pituitary | 0.91 | Excellent — close to Run 5 best (0.93) |
| Background → Glioma | 0.47 FP rate | Dramatic reduction from 0.87 (Run 5) and 0.68 (Run 4). Healthy brain negative samples working as intended. |

### Root Cause
Early stop at epoch 30 — patience=15 too aggressive for a new dataset. cls_loss spiked severely at epochs 2–3 (values 10–12), model stabilized at epoch 4 and was still improving at epoch 30. Val-train divergence near zero (0.024) confirms healthy generalization with no overfitting; patience fired on a transient plateau, not a real convergence.

### Conclusion & Next Run Plan
BRISC confirmed as the correct direction — meningioma generalization bottleneck and background→glioma FP rate both resolved in a single dataset switch. Global metrics are lower than Run 5 only because the model needed more epochs. Run 7: identical config with patience=30 to allow longer exploration past transient plateaus.

---

## Run 7 — yolo11s_11_05_1621

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9195 |
| mAP@0.5:0.95 | 0.5907 |
| Precision | 0.9178 |
| Recall | 0.8840 |

### Run-specific Changes
patience increased 15→30. Full 100 epochs, no early stop. Best epoch: 62/100. BRISC 2025 dataset.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Meningioma | 0.94 | Maintained Run 6 best-ever result |
| Glioma | 0.85 | +0.09 vs Run 6 — 70 extra epochs enabled better learning of the hardest class |
| Pituitary | 0.96 | Best absolute result of any class across all runs in the project |
| Background → Glioma | 0.50 FP rate | Slight regression vs Run 6 (0.47) — acceptable given glioma gain |

### Root Cause
cls_val spiked at epochs 2–3 (values 10–12) — characteristic of BRISC multi-plane multi-protocol variance. Model stabilized at epoch 4. cos_lr produced exemplary terminal convergence: std 0.0027 over last 20 epochs. Val-train divergence 0.184 — healthy generalization. patience=30 was the correct fix for Run 6's premature stop.

### Conclusion & Next Run Plan
Run 7 is the best model of the project. Meningioma, glioma, and pituitary all improved vs the Figshare baseline. Background→Glioma FP rate reduced from 0.87 (Run 5 Figshare) to 0.50. Next: implement False Positive Rate metric on healthy images in `evaluate.py` to properly quantify background suppression from BRISC negative samples.
