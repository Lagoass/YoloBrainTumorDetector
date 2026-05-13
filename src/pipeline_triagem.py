"""Triage pipeline: train → evaluate → healthy_fpr → predict.

Trains a single YOLOv11s binary model (tumor / no_tumor) on
data/triagem_dataset/ — assumes the dataset already exists.

Usage:
  python src/pipeline_triagem.py
  python src/pipeline_triagem.py --epochs 50
"""
import argparse
import sys
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TRIAGEM_ROOT = ROOT / "data" / "triagem_dataset"
TRIAGEM_YAML = str(TRIAGEM_ROOT / "dataset.yaml")

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


def train_triagem(epochs: int) -> tuple[Path, object]:
    run_name = f"yolo11s_triagem_{time.strftime('%d_%m_%H%M')}"
    project = str(ROOT / "runs" / "triagem")
    print(f"\n{'='*60}")
    print(f"Training triage model | epochs={epochs} | name={run_name}")
    print(f"{'='*60}")
    model = YOLO("yolo11s.pt")
    try:
        model.train(data=TRIAGEM_YAML, epochs=epochs, project=project, name=run_name, **TRAIN_KWARGS)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM with batch=16, retrying with batch=8")
            kw = {**TRAIN_KWARGS, "batch": 8}
            model.train(data=TRIAGEM_YAML, epochs=epochs, project=project, name=run_name, **kw)
        else:
            raise
    weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    return weights, model.trainer


def evaluate_triagem(weights: Path):
    from evaluate import evaluate
    return evaluate(weights, data_yaml=TRIAGEM_YAML)


def healthy_fpr_triagem(weights: Path) -> tuple[int, int, float]:
    labels_test = TRIAGEM_ROOT / "labels" / "test"
    images_test = TRIAGEM_ROOT / "images" / "test"

    healthy_images: list[Path] = []
    for lbl in sorted(labels_test.glob("*.txt")):
        if lbl.stat().st_size == 0:
            for ext in (".jpg", ".jpeg", ".png"):
                img = images_test / (lbl.stem + ext)
                if img.exists():
                    healthy_images.append(img)
                    break

    total = len(healthy_images)
    if total == 0:
        print("\nHealthy FPR: no no_tumor images found in triagem_dataset/labels/test/")
        return 0, 0, 0.0

    print(f"\nHealthy FPR: running inference on {total} no_tumor test images")
    model = YOLO(str(weights))
    results = model.predict(
        source=[str(p) for p in healthy_images],
        conf=0.25,
        imgsz=640,
        verbose=False,
    )
    fp_count = sum(1 for r in results if len(r.boxes) > 0)
    fp_rate = fp_count / total
    print(f"  False positives: {fp_count}/{total}  ({fp_rate:.1%})")
    return total, fp_count, fp_rate


def predict_triagem(weights: Path):
    from predict import predict
    return predict(weights, test_images_dir=TRIAGEM_ROOT / "images" / "test")


def print_summary(
    weights: Path,
    trainer,
    metrics,
    eval_dir,
    pred_dir,
    total_healthy: int,
    fp_count: int,
    fp_rate: float,
    train_time: float,
) -> None:
    best_epoch = getattr(trainer, "best_epoch", None)
    epochs_run = getattr(trainer, "epoch", None)
    max_epochs = getattr(trainer, "epochs", None)
    early_stopped = (epochs_run + 1) < max_epochs if (epochs_run is not None and max_epochs is not None) else None

    map50 = getattr(metrics.box, "map50", None)
    map50_95 = getattr(metrics.box, "map", None)
    precision = getattr(metrics.box, "mp", None)
    recall = getattr(metrics.box, "mr", None)

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    clean_count = total_healthy - fp_count
    clean_pct = clean_count / total_healthy if total_healthy else 0.0

    print("\n" + "=" * 70)
    print("TRIAGE PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Dataset          : Triagem (tumor / no_tumor)  nc=1")
    print(f"  Weights          : {weights}")
    print(f"  Eval dir         : {eval_dir}")
    print(f"  Predictions dir  : {pred_dir}")
    print(f"  Train time       : {train_time/60:.1f} min")
    if best_epoch is not None:
        es_note = ""
        if early_stopped:
            es_note = f"  [early stop @ epoch {epochs_run+1}/{max_epochs}]"
        print(f"  Best epoch       : {best_epoch+1}{es_note}")
    print()
    print(f"  mAP@0.50         : {fmt(map50)}")
    print(f"  mAP@0.5:0.95     : {fmt(map50_95)}")
    print(f"  Precision        : {fmt(precision)}")
    print(f"  Recall           : {fmt(recall)}")
    print()
    print("  Healthy Brain Analysis")
    print(f"    Healthy test images : {total_healthy}")
    print(f"    False positives     : {fp_count}  ({fp_rate:.1%})")
    print(f"    Clean predictions   : {clean_count}  ({clean_pct:.1%})")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage brain tumor YOLO pipeline")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    t0 = time.time()
    weights, trainer = train_triagem(args.epochs)
    train_time = time.time() - t0

    metrics, eval_dir = evaluate_triagem(weights)
    total_healthy, fp_count, fp_rate = healthy_fpr_triagem(weights)
    pred_dir = predict_triagem(weights)

    print_summary(
        weights=weights,
        trainer=trainer,
        metrics=metrics,
        eval_dir=eval_dir,
        pred_dir=pred_dir,
        total_healthy=total_healthy,
        fp_count=fp_count,
        fp_rate=fp_rate,
        train_time=train_time,
    )


if __name__ == "__main__":
    main()
