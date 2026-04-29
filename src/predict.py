import random
import sys
from glob import glob
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES_DIR = ROOT / "data" / "dataset" / "images" / "test"
DEFAULT_WEIGHTS = ROOT / "runs" / "brain_tumor" / "yolo11s_run1" / "weights" / "best.pt"


def find_weights(path: Path) -> Path:
    if path.exists():
        return path
    runs_dir = ROOT / "runs" / "brain_tumor"
    if runs_dir.exists():
        candidates = sorted(runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
    return path


def predict(weights_path: Path, n: int = 10, conf: float = 0.25):
    weights_path = find_weights(weights_path)
    if not weights_path.exists():
        print(f"Error: weights not found at {weights_path}")
        print("Train the model first: python src/train.py")
        sys.exit(1)

    images = glob(str(TEST_IMAGES_DIR / "*.jpg"))
    if not images:
        print(f"Error: no test images found in {TEST_IMAGES_DIR}")
        sys.exit(1)

    sample = random.sample(images, min(n, len(images)))
    print(f"Running inference on {len(sample)} random test images (conf={conf})")
    print(f"Model: {weights_path}")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=sample,
        save=True,
        conf=conf,
        imgsz=640,
        project=str(weights_path.parent.parent),
        name="predict",
    )

    save_dir = Path(results[0].save_dir) if results else ROOT / "runs" / "brain_tumor" / "predict"
    print(f"\nAnnotated images saved to: {save_dir}")
    return results


if __name__ == "__main__":
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    predict(weights)
