import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = str(ROOT / "data" / "dataset" / "dataset.yaml")
DEFAULT_WEIGHTS = ROOT / "runs" / "brain_tumor" / "yolo11s_run1" / "weights" / "best.pt"


def find_weights(path: Path) -> Path:
    if path.exists():
        return path
    # Auto-detect latest run
    runs_dir = ROOT / "runs" / "brain_tumor"
    if runs_dir.exists():
        candidates = sorted(runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
    return path


def evaluate(weights_path: Path, data_yaml: str = DATA_YAML):
    weights_path = find_weights(weights_path)
    if not weights_path.exists():
        print(f"Error: weights not found at {weights_path}")
        print("Train the model first: python src/train.py")
        sys.exit(1)

    print(f"Evaluating model: {weights_path}")
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        batch=16,
        project=str(weights_path.parent.parent),
        name="eval_test",
    )
    save_dir = Path(metrics.save_dir)
    print(f"\nResults saved to: {save_dir}")
    return metrics, save_dir


def healthy_fpr(weights_path: Path, test_images_dir: Path, data_yaml: str = DATA_YAML):
    """Run inference on no_tumor test images and return false positive rate.

    No-tumor images are identified by finding empty label files in
    brisc_dataset/labels/test/ and loading their paired images from test_images_dir.
    A false positive is any image that produces at least one predicted bbox.
    """
    weights_path = find_weights(weights_path)
    labels_test_dir = ROOT / "data" / "brisc_dataset" / "labels" / "test"
    if not labels_test_dir.exists():
        print(f"Warning: labels dir not found: {labels_test_dir}")
        return 0, 0, 0.0

    healthy_images = []
    for lbl in sorted(labels_test_dir.glob("*.txt")):
        if lbl.stat().st_size == 0:
            img = test_images_dir / (lbl.stem + ".jpg")
            if img.exists():
                healthy_images.append(img)

    total_healthy = len(healthy_images)
    if total_healthy == 0:
        print("Warning: no empty label files found — no_tumor images missing from test split")
        return 0, 0, 0.0

    print(f"\nHealthy FPR: running inference on {total_healthy} no_tumor test images")
    model = YOLO(str(weights_path))
    results = model.predict(
        source=[str(p) for p in healthy_images],
        conf=0.25,
        imgsz=640,
        verbose=False,
    )

    fp_count = sum(1 for r in results if len(r.boxes) > 0)
    fp_rate = fp_count / total_healthy
    print(f"  False positives : {fp_count}/{total_healthy}  ({fp_rate:.1%})")
    return total_healthy, fp_count, fp_rate


if __name__ == "__main__":
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    metrics, _ = evaluate(weights)
