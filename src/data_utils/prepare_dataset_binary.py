"""
Dissects brisc_dataset into 3 binary datasets (one per tumor type).

For each tumor in {glioma, meningioma, pituitary}:
  - Copies tumor images + labels (bbox rewritten as class 0)
  - Copies all no_tumor images + empty labels
  - Generates data/dissected_brisc/<tumor>/dataset.yaml with nc=1

Run from project root:
  python src/data_utils/prepare_dataset_binary.py
"""
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
BRISC_LABELS = ROOT / "data" / "brisc_dataset" / "labels"
BRISC_IMAGES = ROOT / "data" / "brisc_dataset" / "images"
OUT_ROOT = ROOT / "data" / "dissected_brisc"

TUMORS = ["glioma", "meningioma", "pituitary"]
# class index in brisc_dataset labels → tumor name
CLASS_IDX = {"0": "meningioma", "1": "glioma", "2": "pituitary"}
SPLITS = ["train", "val", "test"]


def get_class(label_path: Path) -> str:
    """Return tumor name or 'no_tumor' by reading first char of label file."""
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return "no_tumor"
    first_char = content[0]
    return CLASS_IDX.get(first_char, "no_tumor")


def main():
    # counts[tumor][split] = {"tumor": int, "no_tumor": int}
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

            for lbl_path in sorted(lbl_src_dir.glob("*.txt")):
                img_path = img_src_dir / (lbl_path.stem + ".jpg")
                if not img_path.exists():
                    continue

                cls = get_class(lbl_path)
                dst_lbl = lbl_dst_dir / lbl_path.name
                dst_img = img_dst_dir / img_path.name

                if cls == tumor:
                    # Rewrite label with class index 0
                    content = lbl_path.read_text(encoding="utf-8").strip()
                    # Replace leading class index with 0
                    parts = content.split(None, 1)
                    rewritten = "0 " + parts[1] + "\n" if len(parts) == 2 else ""
                    dst_lbl.write_text(rewritten, encoding="utf-8")
                    shutil.copy2(img_path, dst_img)
                    counts[tumor][split]["tumor"] += 1
                elif cls == "no_tumor":
                    shutil.copy2(lbl_path, dst_lbl)
                    shutil.copy2(img_path, dst_img)
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
        yaml_path = OUT_ROOT / tumor / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # Summary table
    print(f"\n{'Tumor':<14} {'Split':<6} {'Tumor':>7} {'No-tumor':>9} {'Total':>7}")
    print("-" * 48)
    for tumor in TUMORS:
        for split in SPLITS:
            t = counts[tumor][split]["tumor"]
            n = counts[tumor][split]["no_tumor"]
            print(f"{tumor:<14} {split:<6} {t:>7} {n:>9} {t+n:>7}")
        print()

    print(f"Datasets written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
