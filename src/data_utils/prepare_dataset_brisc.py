"""
Prepares BRISC 2025 dataset for YOLO detection training.

Sources:
  - segmentation_task images (glioma/meningioma/pituitary): mask PNG → YOLO bbox
  - classification_task no_tumor images: empty label (negative examples)

Output: data/brisc_dataset/{images,labels}/{all,train,val,test}/
        data/brisc_dataset/dataset.yaml
"""
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
BRISC_ROOT = ROOT / "data" / "raw_brisc" / "brisc2025"
MANIFEST_CSV = BRISC_ROOT / "manifest.csv"
OUT_ROOT = ROOT / "data" / "brisc_dataset"

# Must match figshare class indices for cross-dataset consistency
CLASS_MAP = {"meningioma": 0, "glioma": 1, "pituitary": 2}

SEED = 42


def mask_to_yolo_bbox(mask_path: Path, img_w: int, img_h: int) -> str | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def normalize_and_save(src: Path, dst: Path) -> None:
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"Cannot read image: {src}")
    img_f = img.astype(np.float32)
    lo, hi = img_f.min(), img_f.max()
    if hi > lo:
        img_f = (img_f - lo) / (hi - lo) * 255.0
    else:
        img_f[:] = 0.0
    cv2.imwrite(str(dst), img_f.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])


def copy_split(samples: list, split_name: str) -> None:
    img_dir = OUT_ROOT / "images" / split_name
    lbl_dir = OUT_ROOT / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for stem, _cls, _plane in samples:
        shutil.copy2(OUT_ROOT / "images" / "all" / (stem + ".jpg"), img_dir / (stem + ".jpg"))
        shutil.copy2(OUT_ROOT / "labels" / "all" / (stem + ".txt"), lbl_dir / (stem + ".txt"))


def main():
    random.seed(SEED)

    for sub in ("images/all", "labels/all"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    seg_rows = [r for r in rows if r["task"] == "segmentation" and r["is_mask"] == "False"]
    notumor_rows = [r for r in rows if r["task"] == "classification" and r["tumor_label"] == "no_tumor"]

    print(f"Segmentation images (tumor classes): {len(seg_rows)}")
    print(f"No-tumor images (classification):    {len(notumor_rows)}")

    samples: list[tuple[str, str, str]] = []  # (stem, class_label, plane_label)
    empty_mask_count = 0

    # --- Tumor images: derive bbox from paired mask ---
    for r in seg_rows:
        rel = Path(r["relative_path"].replace("\\", "/"))
        src_img = BRISC_ROOT / rel
        stem = Path(r["filename"]).stem
        split_orig = r["split"]
        mask_path = BRISC_ROOT / "segmentation_task" / split_orig / "masks" / (stem + ".png")

        out_img = OUT_ROOT / "images" / "all" / (stem + ".jpg")
        normalize_and_save(src_img, out_img)

        cls_idx = CLASS_MAP[r["tumor_label"]]
        img_w, img_h = int(r["width"]), int(r["height"])
        bbox_str = mask_to_yolo_bbox(mask_path, img_w, img_h)
        lbl_path = OUT_ROOT / "labels" / "all" / (stem + ".txt")
        if bbox_str:
            lbl_path.write_text(f"{cls_idx} {bbox_str}\n", encoding="utf-8")
        else:
            lbl_path.write_text("", encoding="utf-8")
            empty_mask_count += 1

        samples.append((stem, r["tumor_label"], r["plane_label"]))

    if empty_mask_count:
        print(f"Warning: {empty_mask_count} masks had no white pixels — written as empty labels")

    # --- No-tumor images: empty label (negative examples) ---
    for r in notumor_rows:
        rel = Path(r["relative_path"].replace("\\", "/"))
        src_img = BRISC_ROOT / rel
        stem = Path(r["filename"]).stem

        out_img = OUT_ROOT / "images" / "all" / (stem + ".jpg")
        normalize_and_save(src_img, out_img)

        lbl_path = OUT_ROOT / "labels" / "all" / (stem + ".txt")
        lbl_path.write_text("", encoding="utf-8")

        samples.append((stem, "no_tumor", r["plane_label"]))

    # --- Stratified 80/10/10 split by class+plane ---
    by_stratum: dict[str, list] = defaultdict(list)
    for s in samples:
        by_stratum[f"{s[1]}_{s[2]}"].append(s)

    train_samples, val_samples, test_samples = [], [], []
    for _key, items in sorted(by_stratum.items()):
        random.shuffle(items)
        n = len(items)
        n_val = max(1, round(n * 0.10))
        n_test = max(1, round(n * 0.10))
        n_train = n - n_val - n_test
        train_samples.extend(items[:n_train])
        val_samples.extend(items[n_train:n_train + n_val])
        test_samples.extend(items[n_train + n_val:])

    copy_split(train_samples, "train")
    copy_split(val_samples, "val")
    copy_split(test_samples, "test")

    # --- dataset.yaml ---
    yaml_data = {
        "path": str(OUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": {0: "meningioma", 1: "glioma", 2: "pituitary tumor"},
    }
    yaml_path = OUT_ROOT / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # --- Summary ---
    print(f"\nTotal: {len(samples)}  |  Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}")

    print("\nClass distribution (all):")
    class_counts: dict[str, int] = defaultdict(int)
    for _, cls, _ in samples:
        class_counts[cls] += 1
    for cls in sorted(class_counts):
        print(f"  {cls:<20} {class_counts[cls]:>5}")

    print("\nPlane distribution (all):")
    plane_counts: dict[str, int] = defaultdict(int)
    for _, _, plane in samples:
        plane_counts[plane] += 1
    for plane in sorted(plane_counts):
        print(f"  {plane:<12} {plane_counts[plane]:>5}")

    print("\nClass+Plane stratified split:")
    train_strat: dict[str, int] = defaultdict(int)
    val_strat: dict[str, int] = defaultdict(int)
    test_strat: dict[str, int] = defaultdict(int)
    for _, cls, plane in train_samples:
        train_strat[f"{cls}_{plane}"] += 1
    for _, cls, plane in val_samples:
        val_strat[f"{cls}_{plane}"] += 1
    for _, cls, plane in test_samples:
        test_strat[f"{cls}_{plane}"] += 1
    all_keys = sorted(set(list(train_strat) + list(val_strat) + list(test_strat)))
    print(f"  {'Stratum':<30} {'Train':>6} {'Val':>6} {'Test':>6}")
    for k in all_keys:
        print(f"  {k:<30} {train_strat.get(k, 0):>6} {val_strat.get(k, 0):>6} {test_strat.get(k, 0):>6}")

    print(f"\nDataset written to : {OUT_ROOT}")
    print(f"YAML               : {yaml_path}")


if __name__ == "__main__":
    main()
