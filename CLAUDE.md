# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Detect brain tumors (meningioma, glioma, pituitary) in MRI via bounding boxes using YOLOv11s. Constrained to 2D/2.5D inference on NVIDIA RTX 5060 (8GB VRAM).

## Commands

All scripts must be run from the project root (`BrainTumorYolo/`).

```bash
# Download raw dataset (~800MB, 3064 .mat files)
python src/download_dataset.py

# Parse .mat files and generate 2.5D JPGs + YOLO labels into data/dataset/*/all/
python src/prepare_dataset.py

# Inspect raw .mat structure (first 5 files)
python src/inspect_mat.py

# Visual sanity-check: bbox alignment on sample MRI (outputs test_alignment*.png)
python src/test_alignment.py
```

## Architecture & Data Pipeline

**Raw format:** MATLAB v7.3 HDF5 `.mat` files with struct `cjdata` containing `label` (1-3), `PID` (patient string), `image` (float array, stored column-major), and `tumorBorder` (flat polygon `[x1,y1,x2,y2,...]`).

**HDF5 transpose:** `cjdata['image'][:]` from h5py is column-major; `.T` is mandatory before any cv2 operation.

**2.5D stacking:** Files are grouped by PID and sorted by numeric filename index (which correlates to Z-axis order). Each output image is a 3-channel JPG where channels map to `[Z-1, Z, Z+1]` (boundary slices duplicate edge). Ground truth bbox belongs exclusively to the center slice Z.

**Class mapping:** `label - 1` → `0=meningioma, 1=glioma, 2=pituitary tumor`.

**bbox conversion:** `tumorBorder` polygon → `cv2.boundingRect` bounding box → YOLO normalized `[x_center, y_center, width, height]`.

## YOLO Training Constraints (never change)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `imgsz` | 640 | Minimum to detect small tumors |
| `batch` | 16 | Hard VRAM ceiling at this resolution |
| `amp` | True | FP16 cuts VRAM 50%, enables Ada Lovelace Tensor Cores |

## Medical Augmentation Policy

| Parameter | Value |
|-----------|-------|
| `mixup` | 0.0 — disabled; fuses anatomically impossible images |
| `flipud` | 0.0 — disabled; destroys dorso-ventral orientation |
| `degrees` | ±10° max — emulates scanner misalignment only |
| `fliplr` | 0.5 — safe; brain has bilateral symmetry |
| `mosaic` | 1.0 with `close_mosaic=10` — disable last 10 epochs |

## Dataset Split Rule

Split **by PID**, not by file. A single patient has multiple slices; mixing them across train/val causes data leakage (model memorizes skull shape). Use `cvind.mat` or a PID-stratified random split (80/10/10).

## Inference Export

```bash
yolo export model=runs/detect/train/weights/best.pt format=engine half=True imgsz=640 workspace=4
```

TensorRT `.engine` fuses layers in C++, operates FP16, latency ~5-6ms vs ~12ms for `.pt`.
