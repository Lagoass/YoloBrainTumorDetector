"""Binary pipeline: train → evaluate → healthy_fpr → predict for each tumor type.

Assumes data/dissected_brisc/ already exists (run prepare_dataset_binary.py first).

Usage:
  python src/pipeline_binary.py                  # all 3 tumors, 100 epochs
  python src/pipeline_binary.py --tumor glioma   # single tumor
  python src/pipeline_binary.py --epochs 1       # quick smoke-test
"""
import argparse
import sys
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TUMORS = ["glioma", "meningioma", "pituitary"]
DISSECTED_ROOT = ROOT / "data" / "dissected_brisc"


def yaml_for(tumor: str) -> str:
    return str(DISSECTED_ROOT / tumor / "dataset.yaml")


BRISC_GLIOMA_YAML = yaml_for("glioma")
BRISC_MENINGIOMA_YAML = yaml_for("meningioma")
BRISC_PITUITARY_YAML = yaml_for("pituitary")

TRAIN_KWARGS = dict(
    imgsz=640,
    batch=16,
    amp=True,
    workers=4,
    patience=30,
    cos_lr=True,
    fliplr=0.5,
    flipud=0.0,
    degrees=10.0,
    mosaic=0.5,
    mixup=0.0,
    hsv_v=0.2,
    hsv_s=0.0,
    close_mosaic=10,
)


def train_binary(tumor: str, epochs: int) -> tuple[Path, object]:
    run_name = f"yolo11s_{tumor}_{time.strftime('%d_%m_%H%M')}"
    project = str(ROOT / "runs" / "binary_brain_tumor" / tumor)
    data_yaml = yaml_for(tumor)
    print(f"\n{'='*60}")
    print(f"Training binary model: {tumor} | epochs={epochs} | name={run_name}")
    print(f"{'='*60}")
    model = YOLO("yolo11s.pt")
    try:
        model.train(data=data_yaml, epochs=epochs, project=project, name=run_name, **TRAIN_KWARGS)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM with batch=16, retrying with batch=8")
            kw = {**TRAIN_KWARGS, "batch": 8}
            model.train(data=data_yaml, epochs=epochs, project=project, name=run_name, **kw)
        else:
            raise
    weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    return weights, model.trainer


def evaluate_binary(weights: Path, tumor: str):
    from evaluate import evaluate
    metrics, save_dir = evaluate(weights, data_yaml=yaml_for(tumor))
    return metrics, save_dir


def healthy_fpr_binary(weights: Path, tumor: str):
    from evaluate import healthy_fpr
    test_images_dir = DISSECTED_ROOT / tumor / "images" / "test"
    # Identify no_tumor images from dissected labels
    labels_test = DISSECTED_ROOT / tumor / "labels" / "test"
    healthy_images = []
    for lbl in sorted(labels_test.glob("*.txt")):
        if lbl.stat().st_size == 0:
            img = test_images_dir / (lbl.stem + ".jpg")
            if img.exists():
                healthy_images.append(img)
    total = len(healthy_images)
    if total == 0:
        return 0, 0, 0.0
    print(f"\nHealthy FPR ({tumor}): running inference on {total} no_tumor test images")
    model = YOLO(str(weights))
    results = model.predict(
        source=[str(p) for p in healthy_images],
        conf=0.25,
        imgsz=640,
        verbose=False,
    )
    fp_count = sum(1 for r in results if len(r.boxes) > 0)
    fp_rate = fp_count / total
    print(f"  False positives : {fp_count}/{total}  ({fp_rate:.1%})")
    return total, fp_count, fp_rate


def predict_binary(weights: Path, tumor: str):
    from predict import predict
    test_images_dir = DISSECTED_ROOT / tumor / "images" / "test"
    return predict(weights, test_images_dir=test_images_dir)


def run_tumor(tumor: str, epochs: int) -> dict:
    weights, trainer = train_binary(tumor, epochs)
    metrics, eval_dir = evaluate_binary(weights, tumor)
    total_healthy, fp_count, fp_rate = healthy_fpr_binary(weights, tumor)
    predict_binary(weights, tumor)

    best_epoch = getattr(trainer, "best_epoch", None)
    epochs_run = getattr(trainer, "epoch", None)
    max_epochs = getattr(trainer, "epochs", epochs)
    early_stopped = (epochs_run + 1) < max_epochs if epochs_run is not None else None

    return {
        "tumor": tumor,
        "weights": weights,
        "map50": getattr(metrics.box, "map50", None),
        "map50_95": getattr(metrics.box, "map", None),
        "precision": getattr(metrics.box, "mp", None),
        "recall": getattr(metrics.box, "mr", None),
        "fpr": fp_rate,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "epochs_run": epochs_run,
        "max_epochs": max_epochs,
    }


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("BINARY PIPELINE SUMMARY")
    print("=" * 80)
    header = f"{'Model':<14} {'mAP@0.50':>10} {'mAP@0.5:0.95':>13} {'Precision':>10} {'Recall':>8} {'FPR healthy':>12}"
    print(header)
    print("-" * 80)
    for r in results:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A  "
        print(
            f"{r['tumor']:<14} {fmt(r['map50']):>10} {fmt(r['map50_95']):>13} "
            f"{fmt(r['precision']):>10} {fmt(r['recall']):>8} {fmt(r['fpr']):>12}"
        )
    print("=" * 80)
    for r in results:
        if r["best_epoch"] is not None:
            es = ""
            if r["early_stopped"]:
                es = f"  [early stop @ epoch {r['epochs_run']+1}/{r['max_epochs']}]"
            print(f"  {r['tumor']}: best epoch {r['best_epoch']+1}{es}")
            print(f"    weights: {r['weights']}")


def main():
    parser = argparse.ArgumentParser(description="Binary brain tumor YOLO pipeline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--tumor", choices=TUMORS, default=None, help="Run only one tumor model")
    args = parser.parse_args()

    targets = [args.tumor] if args.tumor else TUMORS
    results = []
    for tumor in targets:
        r = run_tumor(tumor, args.epochs)
        results.append(r)

    print_summary(results)


if __name__ == "__main__":
    main()
