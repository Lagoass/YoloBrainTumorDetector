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

## DIRECTORY STRUCTURE
```text
BrainTumorYolo/
├── data/
│   ├── raw/                 # 3064 .mat + cvind.mat
│   └── dataset/
│       ├── dataset.yaml     # YOLO dataset config (absolute path, nc=3)
│       ├── images/
│       │   ├── all/         # 3064 2.5D .jpg (source, kept intact)
│       │   ├── train/       # 2406 slices (186 PIDs)
│       │   ├── val/         #  373 slices ( 23 PIDs)
│       │   └── test/        #  285 slices ( 24 PIDs)
│       └── labels/
│           ├── all/         # 3064 YOLO .txt (source, kept intact)
│           ├── train/
│           ├── val/
│           └── test/
├── informative/             
│   ├── escopo.md            # Macro goals
│   ├── content.md           # Current LLM context
│   ├── bigAnalisis.txt      # Source analysis
│   └── eval_plan.md         # Implementation plan for evaluation & inference
├── src/
│   ├── download_dataset.py  # Figshare API fetcher
│   ├── prepare_dataset.py   # .mat parser, bbox calc, 2.5D stacker
│   ├── split_dataset.py     # PID-based train/val/test split + dataset.yaml gen
│   ├── train.py             # YOLOv11s training script (GPU 8GB optimized, medical augs)
│   ├── evaluate.py          # Formal evaluation: mAP/P/R on test split via model.val()
│   ├── predict.py           # Visual inference: 10 random test images, saves annotated JPGs
│   ├── inspect_mat.py       # Helper: MATLAB structure check
│   └── test_alignment.py    # Helper: bbox visual alignment
├── run.md                   # Cheatsheet: Conda GPU setup and execution commands
└── CLAUDE.md                # Claude Code context
```

## CURRENT STATUS
- [X] Download & extraction complete.
- [X] Data parsed and formatted. All 3064 files in `data/dataset/*/all`.
- [X] Train/Val/Test split by PID (seed=42, 80/10/10). `src/split_dataset.py`.
- [X] `data/dataset/dataset.yaml` generated (absolute path, nc=3).
- [X] Local GPU Setup (Miniconda, Python 3.12, CUDA 12.1). Cheatsheet in `run.md`.
- [X] `src/train.py` written and verified running on RTX 5060 (imgsz=640, batch=16, amp=True). Augs: degrees=10.0, hsv_s=0.0.
- [X] `src/evaluate.py` implemented: runs `model.val(split="test")`, auto-detects latest best.pt, saves metrics to `runs/brain_tumor/eval_test`.
- [X] `src/predict.py` implemented: samples 10 random test images, runs `model.predict(conf=0.25)`, saves annotated JPGs to `runs/brain_tumor/predict`.

## NEXT STEPS
1. Run training: `python src/train.py`
2. Evaluate: `python src/evaluate.py`
3. Inspect predictions: `python src/predict.py`
4. Export to TensorRT: `yolo export model=runs/brain_tumor/yolo11s_run1/weights/best.pt format=engine half=True imgsz=640 workspace=4`
