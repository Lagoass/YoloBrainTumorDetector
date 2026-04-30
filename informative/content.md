# PROJECT STATE: Brain Tumor YOLO

## STACK & CONSTRAINTS
- Target Model: YOLOv11s
- Env: Python, h5py, opencv-python, numpy
- Hardware Limit: 8GB VRAM

## ARCHITECTURE & DATA PIPELINE
- Source Data: 3064 MATLAB v7.3 `.mat` files (MRI 512x512, `tumorBorder` vector)
- Target Format: YOLO bounding boxes `[class x_center y_center width height]` normalized.
- Classes: `0: meningioma`, `1: glioma`, `2: pituitary tumor`
- Stacking Strategy: 2.5D RGB simulated via adjacent Z-slices (Z-1, Z, Z+1) grouped by `PID`. Matrix transposition applied for HDF5 to OpenCV alignment.

### Dataset Class Distribution
| Class | Slices | % |
|---|---|---|
| Meningioma | 708 | 23.1% |
| Glioma | 1426 | 46.5% |
| Pituitary tumor | 930 | 30.4% |
| **Total** | **3064** | 233 patients |

> Glioma is nearly 2× more represented than meningioma. This imbalance is the confirmed root cause of glioma overconfidence observed in Run 1.

## DIRECTORY STRUCTURE
```text
BrainTumorYolo/
├── data/
│   ├── raw/                 # 3064 .mat + cvind.mat
│   ├── dataset/
│   │   ├── dataset.yaml     # YOLO dataset config (absolute path, nc=3)
│   │   ├── images/
│   │   │   ├── all/         # 3064 2.5D .jpg (source, kept intact)
│   │   │   ├── train/       # 2406 slices (186 PIDs)
│   │   │   ├── val/         #  373 slices ( 23 PIDs)
│   │   │   └── test/        #  285 slices ( 24 PIDs)
│   │   └── labels/
│   │       ├── all/         # 3064 YOLO .txt (source, kept intact)
│   │       ├── train/       # meningioma=552, glioma=1087, pituitary=767
│   │       ├── val/
│   │       └── test/
│   └── oversample_dataset/
│       ├── dataset.yaml     # YOLO config pointing to oversample_dataset (absolute path)
│       ├── images/
│       │   ├── train/       # 3261 slices (meningioma=1087, glioma=1087, pituitary=1087)
│       │   ├── val/         #  373 slices (unchanged)
│       │   └── test/        #  285 slices (unchanged)
│       └── labels/
│           ├── train/       # 3261 YOLO .txt (oversampled, _os1/_os2 suffixes on dupes)
│           ├── val/
│           └── test/
├── informative/             
│   ├── escopo.md                # Macro goals
│   ├── content.md               # Current LLM context
│   ├── bigAnalisis.txt          # Source analysis
│   ├── eval_plan.md             # Implementation plan for evaluation & inference
│   ├── run_analysis.md          # Scientific log: metrics + findings per training run
│   └── analysis_notebook.ipynb  # Interactive analysis: learning curves, loss, confusion matrices, F1
├── src/
│   ├── data_utils/
│   │   ├── __init__.py
│   │   ├── download_dataset.py  # Figshare API fetcher
│   │   ├── prepare_dataset.py   # .mat parser, bbox calc, 2.5D stacker
│   │   ├── split_dataset.py     # PID-based train/val/test split + dataset.yaml gen
│   │   ├── oversample.py        # Oversamples meningioma+pituitary to glioma parity (seed=42)
│   │   ├── inspect_mat.py       # Helper: MATLAB structure check
│   │   ├── inspect_mat2.py      # Helper: MATLAB structure check (variant)
│   │   └── test_alignment.py    # Helper: bbox visual alignment
│   ├── train.py             # YOLOv11s training script (GPU 8GB optimized, medical augs)
│   ├── evaluate.py          # Formal evaluation: mAP/P/R on test split via model.val()
│   ├── predict.py           # Visual inference: 10 random test images, saves annotated JPGs
│   └── pipeline.py          # Full orchestrator: train → evaluate → predict (CLI)
├── run.md                   # Cheatsheet: Conda GPU setup and execution commands
└── CLAUDE.md                # Claude Code context
```

## CURRENT STATUS
- [X] Download & extraction complete.
- [X] Data parsed and formatted. All 3064 files in `data/dataset/*/all`.
- [X] Train/Val/Test split by PID (seed=42, 80/10/10). `src/split_dataset.py`.
- [X] `data/dataset/dataset.yaml` generated (absolute path, nc=3).
- [X] Local GPU Setup (Miniconda, Python 3.12, CUDA 12.1). Cheatsheet in `run.md`.
- [X] `src/train.py` written and verified running on RTX 5060 (imgsz=640, batch=16, amp=True). Augs: degrees=10.0, hsv_s=0.0. `cos_lr=True`, `patience=15`. Run name auto-generated as `yolo11s_DD_MM_HHMM`. Accepts `data_yaml` parameter (defaults to `OVERSAMPLE_YAML`); constants `DATA_YAML` and `OVERSAMPLE_YAML` defined at module level.
- [X] `src/evaluate.py` implemented: runs `model.val(split="test")`, auto-detects latest best.pt, saves metrics alongside the run (`<run_dir>/eval_test`). Uses `metrics.save_dir` (not `model.validator.save_dir`) and `project=weights_path.parent.parent`.
- [X] `src/predict.py` implemented: samples 10 random test images, runs `model.predict(conf=0.25)`, saves annotated JPGs to `<run_dir>/predict`. Uses `project=weights_path.parent.parent`.
- [X] `src/oversample.py` implemented: copies full dataset to `data/oversample_dataset/`, then duplicates meningioma (552→1087) and pituitary (767→1087) train samples via `random.choices(pool, k=needed)` with `seed=42`. Each duplicate gets `_os1`, `_os2`, ... suffix. Final train: 3261 balanced images. Generates `oversample_dataset/dataset.yaml`.
- [X] `informative/analysis_notebook.ipynb` created: 7-section interactive analysis covering run config, learning curves, loss curves, convergence table, global metrics bar chart, confusion matrices, and F1-confidence curves. Auto-detects runs from `runs/brain_tumor/*/results.csv`.

## TRAINING RUNS
- **Run 1 (`yolo11s_29_04_1620`):** mAP@0.50=0.9217, mAP@0.5:0.95=0.5468, P=0.9244, R=0.8453. Root cause: glioma class imbalance causing overconfidence. See `informative/run_analysis.md`.
- **Run 2 (`yolo11s_29_04_1759`):** mAP@0.50=0.9217, mAP@0.5:0.95=0.5468, P=0.9244, R=0.8453. `label_smoothing=0.1` tested — had no effect (symmetric operation, insensitive to class distribution). Identical trajectory to Run 1.
- **Run 3 (`yolo11s_29_04_2007`):** mAP@0.50=0.9290, mAP@0.5:0.95=0.6212, P=0.9301, R=0.8844. Oversampling confirmed effective.

## NEXT STEPS
1. Run 4: `python src/pipeline.py` — cos_lr=True + patience=15 applied. Expected: later best epoch, less oscillation, potential mAP@0.5:0.95 improvement.
2. Export to TensorRT: `yolo export model=runs/brain_tumor/<run_name>/weights/best.pt format=engine half=True imgsz=640 workspace=4`
