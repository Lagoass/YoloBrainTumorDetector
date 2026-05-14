"""
Generates 4 binary datasets simultaneously, one per negative sample ratio:
10%, 20%, 30%, 40%.

Output: data/dissected_brisc_ratios/{10,20,30,40}pct/{glioma,meningioma,pituitary}/
  dataset.yaml  (nc=1, absolute path, names: {0: tumor_name})
  images/{train,val,test}/
  labels/{train,val,test}/

Sampling formula per (ratio, tumor, split):
  n_notumor = round(n_tumor * RATIO / (1 - RATIO))
  sampled = rng.sample(pool, min(n_notumor, len(pool)))
  One shared random.Random(SEED=42) across all iterations.

Run from project root:
  python src/data_utils/prepare_dataset_binary_ratios.py
"""
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
BRISC_LABELS = ROOT / "data" / "brisc_dataset" / "labels"
BRISC_IMAGES = ROOT / "data" / "brisc_dataset" / "images"
OUT_ROOT = ROOT / "data" / "dissected_brisc_ratios"

TUMORS = ["glioma", "meningioma", "pituitary"]
CLASS_IDX = {"0": "meningioma", "1": "glioma", "2": "pituitary"}
SPLITS = ["train", "val", "test"]
RATIOS = [10, 20, 30, 40]
SEED = 42


def get_class(label_path: Path) -> str:
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return "no_tumor"
    return CLASS_IDX.get(content[0], "no_tumor")


def main():
    rng = random.Random(SEED)

    # counts[ratio][tumor][split] = {"tumor": int, "no_tumor": int}
    counts: dict[int, dict[str, dict[str, dict[str, int]]]] = {
        r: {t: {s: {"tumor": 0, "no_tumor": 0} for s in SPLITS} for t in TUMORS}
        for r in RATIOS
    }

    # Pre-bucket all label paths by (tumor/no_tumor, split) — read once
    tumor_lbls: dict[str, dict[str, list[Path]]] = {t: {s: [] for s in SPLITS} for t in TUMORS}
    notumor_lbls: dict[str, list[Path]] = {s: [] for s in SPLITS}

    for split in SPLITS:
        lbl_src_dir = BRISC_LABELS / split
        img_src_dir = BRISC_IMAGES / split
        for lbl_path in sorted(lbl_src_dir.glob("*.txt")):
            img_path = img_src_dir / (lbl_path.stem + ".jpg")
            if not img_path.exists():
                continue
            cls = get_class(lbl_path)
            if cls in TUMORS:
                tumor_lbls[cls][split].append(lbl_path)
            elif cls == "no_tumor":
                notumor_lbls[split].append(lbl_path)

    for ratio in RATIOS:
        ratio_root = OUT_ROOT / f"{ratio}pct"
        frac = ratio / 100.0

        for tumor in TUMORS:
            for split in SPLITS:
                img_src_dir = BRISC_IMAGES / split
                lbl_dst_dir = ratio_root / tumor / "labels" / split
                img_dst_dir = ratio_root / tumor / "images" / split
                lbl_dst_dir.mkdir(parents=True, exist_ok=True)
                img_dst_dir.mkdir(parents=True, exist_ok=True)

                # Copy tumor images with class index rewritten to 0
                for lbl_path in tumor_lbls[tumor][split]:
                    img_path = img_src_dir / (lbl_path.stem + ".jpg")
                    content = lbl_path.read_text(encoding="utf-8").strip()
                    parts = content.split(None, 1)
                    rewritten = "0 " + parts[1] + "\n" if len(parts) == 2 else ""
                    (lbl_dst_dir / lbl_path.name).write_text(rewritten, encoding="utf-8")
                    shutil.copy2(img_path, img_dst_dir / img_path.name)
                    counts[ratio][tumor][split]["tumor"] += 1

                # Sample proportional no_tumor subset
                n_tumor = len(tumor_lbls[tumor][split])
                n_notumor = round(n_tumor * frac / (1 - frac))
                pool = notumor_lbls[split]
                sampled = rng.sample(pool, min(n_notumor, len(pool)))

                for lbl_path in sampled:
                    img_path = img_src_dir / (lbl_path.stem + ".jpg")
                    shutil.copy2(lbl_path, lbl_dst_dir / lbl_path.name)
                    shutil.copy2(img_path, img_dst_dir / img_path.name)
                    counts[ratio][tumor][split]["no_tumor"] += 1

            # dataset.yaml per tumor
            yaml_data = {
                "path": str(ratio_root / tumor),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 1,
                "names": {0: tumor},
            }
            with open(ratio_root / tumor / "dataset.yaml", "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    # Summary table
    col = f"{'Ratio':<6} {'Tumor':<14} {'Split':<6} {'Tumor':>7} {'No-tumor':>9} {'Total':>7} {'Actual%':>8}"
    print(f"\n{col}")
    print("-" * 64)
    for ratio in RATIOS:
        for tumor in TUMORS:
            for split in SPLITS:
                t = counts[ratio][tumor][split]["tumor"]
                n = counts[ratio][tumor][split]["no_tumor"]
                total = t + n
                actual = n / total * 100 if total else 0.0
                print(f"{ratio:<6} {tumor:<14} {split:<6} {t:>7} {n:>9} {total:>7} {actual:>7.1f}%")
            print()

    print(f"Datasets written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
