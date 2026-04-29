#!/usr/bin/env python3
"""
Oversample minority classes in the training split to reach glioma (class 1) parity.
Copies the full dataset structure, appending _os1, _os2, ... suffixes to duplicate files.
Run from project root: python src/oversample.py
"""

import random
import shutil
from pathlib import Path

SEED = 42
DATASET_DIR = Path("data/dataset")
OVERSAMPLE_DIR = Path("data/oversample_dataset")
CLASS_NAMES = {0: "meningioma", 1: "glioma", 2: "pituitary"}


def read_class(label_path):
    with open(label_path) as f:
        return int(f.readline().split()[0])


def collect_train_classes(labels_dir):
    class_files = {0: [], 1: [], 2: []}
    for p in sorted(labels_dir.glob("*.txt")):
        cls = read_class(p)
        class_files[cls].append(p.stem)
    return class_files


def copy_pair(stem, src_img, src_lbl, dst_img, dst_lbl, new_stem=None):
    new_stem = new_stem or stem
    shutil.copy2(src_img / f"{stem}.jpg", dst_img / f"{new_stem}.jpg")
    shutil.copy2(src_lbl / f"{stem}.txt", dst_lbl / f"{new_stem}.txt")


def main():
    random.seed(SEED)

    for split in ("train", "val", "test"):
        (OVERSAMPLE_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OVERSAMPLE_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_lbl_dir = DATASET_DIR / "labels" / "train"
    class_files = collect_train_classes(train_lbl_dir)

    print("Original train class counts:")
    for cls in (0, 1, 2):
        print(f"  {CLASS_NAMES[cls]:18s} (class {cls}): {len(class_files[cls])}")

    glioma_count = len(class_files[1])

    src_img = DATASET_DIR / "images" / "train"
    src_lbl = DATASET_DIR / "labels" / "train"
    dst_img = OVERSAMPLE_DIR / "images" / "train"
    dst_lbl = OVERSAMPLE_DIR / "labels" / "train"

    for stems in class_files.values():
        for stem in stems:
            copy_pair(stem, src_img, src_lbl, dst_img, dst_lbl)

    final_counts = {cls: len(files) for cls, files in class_files.items()}
    for cls in (0, 2):
        pool = class_files[cls]
        needed = glioma_count - len(pool)
        if needed <= 0:
            continue
        copies = random.choices(pool, k=needed)
        stem_counter = {}
        for stem in copies:
            stem_counter[stem] = stem_counter.get(stem, 0) + 1
            new_stem = f"{stem}_os{stem_counter[stem]}"
            copy_pair(stem, src_img, src_lbl, dst_img, dst_lbl, new_stem=new_stem)
        final_counts[cls] += needed

    for split in ("val", "test"):
        for subdir in ("images", "labels"):
            src = DATASET_DIR / subdir / split
            dst = OVERSAMPLE_DIR / subdir / split
            for f in src.iterdir():
                shutil.copy2(f, dst / f.name)

    abs_path = str(OVERSAMPLE_DIR.resolve())
    yaml_content = f"""path: {abs_path}
train: images/train
val:   images/val
test:  images/test

nc: 3
names:
  0: meningioma
  1: glioma
  2: pituitary tumor

# Oversampled train (seed={SEED}):
# meningioma={final_counts[0]}, glioma={final_counts[1]}, pituitary={final_counts[2]}
# val/test: unchanged from data/dataset/
"""
    (OVERSAMPLE_DIR / "dataset.yaml").write_text(yaml_content)

    print("\nFinal train class counts after oversampling:")
    for cls in (0, 1, 2):
        print(f"  {CLASS_NAMES[cls]:18s} (class {cls}): {final_counts[cls]}")
    print(f"  {'Total':18s}          : {sum(final_counts.values())}")
    print(f"\ndataset.yaml → {OVERSAMPLE_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
