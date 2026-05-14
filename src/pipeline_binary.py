"""Binary pipeline: train → evaluate → healthy_fpr → predict for each tumor type.

With --ratio (10/20/30/40): reads from data/dissected_brisc_ratios/{ratio}pct/
and saves to runs/binary_brain_tumor/{ratio}pct/{tumor}/.

With --ratio all: runs all 4 ratios sequentially and prints a consolidated
comparison table at the end.

Usage:
  python src/pipeline_binary.py                        # all tumors, ratio=40, 100 epochs
  python src/pipeline_binary.py --tumor glioma         # single tumor
  python src/pipeline_binary.py --ratio 10             # 10% negatives
  python src/pipeline_binary.py --ratio all            # ablation across 10/20/30/40
  python src/pipeline_binary.py --ratio all --tumor meningioma
  python src/pipeline_binary.py --epochs 1             # quick smoke-test
"""
import argparse
import sys
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TUMORS = ["glioma", "meningioma", "pituitary"]
RATIOS = [10, 20, 30, 40]
RATIOS_ROOT = ROOT / "data" / "dissected_brisc_ratios"

# Kept for backward compatibility (no --ratio flag path)
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


def _ratio_root(ratio: int) -> Path:
    return RATIOS_ROOT / f"{ratio}pct"


def train_binary(tumor: str, epochs: int, ratio: int) -> tuple[Path, object]:
    run_name = f"yolo11s_{tumor}_{time.strftime('%d_%m_%H%M')}"
    data_root = _ratio_root(ratio)
    project = str(ROOT / "runs" / "binary_brain_tumor" / f"{ratio}pct" / tumor)
    data_yaml = str(data_root / tumor / "dataset.yaml")
    print(f"\n{'='*60}")
    print(f"Training binary model: {tumor} | ratio={ratio}% | epochs={epochs} | name={run_name}")
    print(f"Data root: {data_root}")
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


def evaluate_binary(weights: Path, tumor: str, ratio: int):
    from evaluate import evaluate
    data_yaml = str(_ratio_root(ratio) / tumor / "dataset.yaml")
    metrics, save_dir = evaluate(weights, data_yaml=data_yaml)
    return metrics, save_dir


def healthy_fpr_binary(weights: Path, tumor: str, ratio: int):
    data_root = _ratio_root(ratio)
    test_images_dir = data_root / tumor / "images" / "test"
    labels_test = data_root / tumor / "labels" / "test"
    healthy_images = []
    for lbl in sorted(labels_test.glob("*.txt")):
        if lbl.stat().st_size == 0:
            img = test_images_dir / (lbl.stem + ".jpg")
            if img.exists():
                healthy_images.append(img)
    total = len(healthy_images)
    if total == 0:
        return 0, 0, 0.0
    print(f"\nHealthy FPR ({tumor}, ratio={ratio}%): running inference on {total} no_tumor test images")
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


def predict_binary(weights: Path, tumor: str, ratio: int):
    from predict import predict
    test_images_dir = _ratio_root(ratio) / tumor / "images" / "test"
    return predict(weights, test_images_dir=test_images_dir)


def run_tumor(tumor: str, epochs: int, ratio: int) -> dict:
    weights, trainer = train_binary(tumor, epochs, ratio)
    metrics, eval_dir = evaluate_binary(weights, tumor, ratio)
    total_healthy, fp_count, fp_rate = healthy_fpr_binary(weights, tumor, ratio)
    predict_binary(weights, tumor, ratio)

    best_epoch = getattr(trainer, "best_epoch", None)
    epochs_run = getattr(trainer, "epoch", None)
    max_epochs = getattr(trainer, "epochs", epochs)
    early_stopped = (epochs_run + 1) < max_epochs if epochs_run is not None else None

    return {
        "tumor": tumor,
        "ratio": ratio,
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


def fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "  N/A  "


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("BINARY PIPELINE SUMMARY")
    print("=" * 80)
    header = f"{'Model':<14} {'mAP@0.50':>10} {'mAP@0.5:0.95':>13} {'Precision':>10} {'Recall':>8} {'FPR healthy':>12}"
    print(header)
    print("-" * 80)
    for r in results:
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


def print_ratio_summary(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("RATIO ABLATION SUMMARY")
    print("=" * 90)
    header = (
        f"{'Tumor':<14} {'Ratio':>6} {'mAP@0.50':>10} {'mAP@0.5:0.95':>13} "
        f"{'Precision':>10} {'Recall':>8} {'FPR healthy':>12}"
    )
    print(header)
    print("-" * 90)
    for tumor in TUMORS:
        for r in [x for x in results if x["tumor"] == tumor]:
            print(
                f"{r['tumor']:<14} {r['ratio']:>5}% {fmt(r['map50']):>10} {fmt(r['map50_95']):>13} "
                f"{fmt(r['precision']):>10} {fmt(r['recall']):>8} {fmt(r['fpr']):>12}"
            )
        print()
    print("=" * 90)
    for r in results:
        if r["best_epoch"] is not None:
            es = ""
            if r["early_stopped"]:
                es = f"  [early stop @ epoch {r['epochs_run']+1}/{r['max_epochs']}]"
            print(f"  {r['tumor']} ratio={r['ratio']}%: best epoch {r['best_epoch']+1}{es}")
            print(f"    weights: {r['weights']}")


def main():
    parser = argparse.ArgumentParser(description="Binary brain tumor YOLO pipeline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--tumor", choices=TUMORS, default=None, help="Run only one tumor model")
    parser.add_argument(
        "--ratio",
        choices=["10", "20", "30", "40", "all"],
        default="40",
        help="Negative sample ratio in %% (default: 40). Use 'all' for ablation across 10/20/30/40.",
    )
    args = parser.parse_args()

    targets = [args.tumor] if args.tumor else TUMORS

    if args.ratio == "all":
        results = []
        for ratio in RATIOS:
            for tumor in targets:
                r = run_tumor(tumor, args.epochs, ratio)
                results.append(r)
        print_ratio_summary(results)
    else:
        ratio = int(args.ratio)
        results = []
        for tumor in targets:
            r = run_tumor(tumor, args.epochs, ratio)
            results.append(r)
        print_summary(results)


if __name__ == "__main__":
    main()
