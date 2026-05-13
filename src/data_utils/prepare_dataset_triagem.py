"""
Builds data/triagem_dataset/ from data/brisc_dataset/.

Reuses the exact train/val/test split from brisc_dataset — no re-splitting.
- Tumor images  : any non-empty label → class index replaced with 0, bbox kept.
- No-tumor images: empty label → copied as-is.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "brisc_dataset"
DST = ROOT / "data" / "triagem_dataset"

SPLITS = ("train", "val", "test")


def process_split(split: str) -> tuple[int, int]:
    src_labels = SRC / "labels" / split
    src_images = SRC / "images" / split
    dst_labels = DST / "labels" / split
    dst_images = DST / "images" / split
    dst_labels.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)

    tumor_count = 0
    no_tumor_count = 0

    for lbl_path in sorted(src_labels.glob("*.txt")):
        stem = lbl_path.stem
        content = lbl_path.read_text().strip()

        # Copy label
        dst_lbl = dst_labels / lbl_path.name
        if content:
            # Rewrite: replace class index with 0, keep bbox coords
            new_lines = []
            for line in content.splitlines():
                parts = line.split()
                parts[0] = "0"
                new_lines.append(" ".join(parts))
            dst_lbl.write_text("\n".join(new_lines) + "\n")
            tumor_count += 1
        else:
            dst_lbl.write_text("")
            no_tumor_count += 1

        # Copy matching image (jpg or png)
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = src_images / (stem + ext)
            if img_path.exists():
                shutil.copy2(img_path, dst_images / img_path.name)
                break

    return tumor_count, no_tumor_count


def write_yaml(abs_dst: Path) -> None:
    yaml_content = (
        f"path: {abs_dst}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: tumor\n"
    )
    (abs_dst / "dataset.yaml").write_text(yaml_content)


def main() -> None:
    print("Building triagem_dataset from brisc_dataset …\n")

    rows: list[tuple[str, int, int, int]] = []
    for split in SPLITS:
        tumor, no_tumor = process_split(split)
        rows.append((split, tumor, no_tumor, tumor + no_tumor))

    write_yaml(DST)

    # Summary table
    header = f"{'Split':<8} {'Tumor':>8} {'No-tumor':>10} {'Total':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for split, tumor, no_tumor, total in rows:
        print(f"{split:<8} {tumor:>8} {no_tumor:>10} {total:>8}")
    print(sep)
    t_tumor = sum(r[1] for r in rows)
    t_no_tumor = sum(r[2] for r in rows)
    print(f"{'TOTAL':<8} {t_tumor:>8} {t_no_tumor:>10} {t_tumor + t_no_tumor:>8}")
    print(f"\ndataset.yaml written to {DST / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
