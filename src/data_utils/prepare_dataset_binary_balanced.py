"""
Dissects brisc_dataset into 3 binary datasets with balanced negatives.

For each tumor in {glioma, meningioma, pituitary}:
  - Copies tumor images + labels (bbox rewritten as class 0)
  - Samples a proportional subset of no_tumor images to achieve
    TARGET_RATIO = 10% negatives per split
  - Generates data/dissected_brisc_balanced/<tumor>/dataset.yaml with nc=1

Sampling formula per (tumor, split):
  n_notumor = round(n_tumor * TARGET_RATIO / (1 - TARGET_RATIO))
  pool sampled with random.sample(pool, min(n_notumor, len(pool))), seed=42

Run from project root:
  python src/data_utils/prepare_dataset_binary_balanced.py
"""
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
BRISC_LABELS = ROOT / "data" / "brisc_dataset" / "labels"
BRISC_IMAGES = ROOT / "data" / "brisc_dataset" / "images"
OUT_ROOT = ROOT / "data" / "dissected_brisc_balanced"

TUMORS = ["glioma", "meningioma", "pituitary"]
CLASS_IDX = {"0": "meningioma", "1": "glioma", "2": "pituitary"}
SPLITS = ["train", "val", "test"]
TARGET_RATIO = 0.10
SEED = 42


def get_class(label_path: Path) -> str:
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return "no_tumor"
    return CLASS_IDX.get(content[0], "no_tumor")


def main():
    rng = random.Random(SEED)
    counts: dict[str, dict[str, dict[str, int]]] = {
        t: {s: {"tumor": 0, "no_tumor": 0} for s in SPLITS} for t in TUMORS
    }

    for tumor in TUMORS:
        for split in SPLITS:
            lbl_src_dir = BRISC_LABELS / split
            img_src_dir = BRISC_IMAGES / split
            lbl_dst_dir = OUT_ROOT / tumor / "labels" / split
            img_dst_dir = OUT_ROOT / tumor / "images" / split
            lbl_dst_dir.mkdir(parents=True, exist_ok=True)
            img_dst_dir.mkdir(parents=True, exist_ok=True)

            tumor_lbls: list[Path] = []
            notumor_lbls: list[Path] = []

            for lbl_path in sorted(lbl_src_dir.glob("*.txt")):
                img_path = img_src_dir / (lbl_path.stem + ".jpg")
                if not img_path.exists():
                    continue
                cls = get_class(lbl_path)
                if cls == tumor:
                    tumor_lbls.append(lbl_path)
                elif cls == "no_tumor":
                    notumor_lbls.append(lbl_path)

            # Copy tumor images with class index rewritten to 0
            for lbl_path in tumor_lbls:
                img_path = img_src_dir / (lbl_path.stem + ".jpg")
                content = lbl_path.read_text(encoding="utf-8").strip()
                parts = content.split(None, 1)
                rewritten = "0 " + parts[1] + "\n" if len(parts) == 2 else ""
                (lbl_dst_dir / lbl_path.name).write_text(rewritten, encoding="utf-8")
                shutil.copy2(img_path, img_dst_dir / img_path.name)
                counts[tumor][split]["tumor"] += 1

            # Sample proportional no_tumor subset
            n_tumor = len(tumor_lbls)
            n_notumor = round(n_tumor * TARGET_RATIO / (1 - TARGET_RATIO))
            sampled = rng.sample(notumor_lbls, min(n_notumor, len(notumor_lbls)))

            for lbl_path in sampled:
                img_path = img_src_dir / (lbl_path.stem + ".jpg")
                shutil.copy2(lbl_path, lbl_dst_dir / lbl_path.name)
                shutil.copy2(img_path, img_dst_dir / img_path.name)
                counts[tumor][split]["no_tumor"] += 1

        # dataset.yaml
        yaml_data = {
            "path": str(OUT_ROOT / tumor),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": {0: tumor},
        }
        with open(OUT_ROOT / tumor / "dataset.yaml", "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # Summary table
    print(f"\n{'Tumor':<14} {'Split':<6} {'Tumor':>7} {'No-tumor':>9} {'Total':>7} {'Ratio':>7}")
    print("-" * 56)
    for tumor in TUMORS:
        for split in SPLITS:
            t = counts[tumor][split]["tumor"]
            n = counts[tumor][split]["no_tumor"]
            total = t + n
            ratio = n / total if total else 0.0
            print(f"{tumor:<14} {split:<6} {t:>7} {n:>9} {total:>7} {ratio:>7.1%}")
        print()

    print(f"Datasets written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
