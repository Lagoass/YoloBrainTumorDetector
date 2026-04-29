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


def evaluate(weights_path: Path):
    weights_path = find_weights(weights_path)
    if not weights_path.exists():
        print(f"Error: weights not found at {weights_path}")
        print("Train the model first: python src/train.py")
        sys.exit(1)

    print(f"Evaluating model: {weights_path}")
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        batch=16,
        project=str(ROOT / "runs" / "brain_tumor"),
        name="eval_test",
    )
    save_dir = Path(model.validator.save_dir)
    print(f"\nResults saved to: {save_dir}")
    return metrics, save_dir


if __name__ == "__main__":
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    metrics, _ = evaluate(weights)
