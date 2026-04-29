"""Full pipeline: train → evaluate → predict."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def find_latest_weights() -> Path | None:
    runs_dir = ROOT / "runs" / "brain_tumor"
    if runs_dir.exists():
        candidates = sorted(runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor YOLO full pipeline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use latest best.pt")
    args = parser.parse_args()

    weights_path: Path | None = None

    # --- Train ---
    if args.skip_train:
        weights_path = find_latest_weights()
        if weights_path is None:
            print("Error: --skip-train specified but no best.pt found under runs/brain_tumor/")
            sys.exit(1)
        print(f"Skipping training. Using weights: {weights_path}")
    else:
        from train import train
        save_dir = train(epochs=args.epochs)
        weights_path = Path(save_dir) / "weights" / "best.pt"
        print(f"\nTraining complete. Weights: {weights_path}")

    # --- Evaluate ---
    from evaluate import evaluate
    metrics, eval_save_dir = evaluate(weights_path)

    # --- Predict ---
    from predict import predict
    results = predict(weights_path)
    predict_save_dir = Path(results[0].save_dir) if results else ROOT / "runs" / "brain_tumor" / "predict"

    # --- Summary ---
    map50 = getattr(metrics.box, "map50", None)
    map50_95 = getattr(metrics.box, "map", None)
    precision = getattr(metrics.box, "mp", None)
    recall = getattr(metrics.box, "mr", None)

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Weights     : {weights_path}")
    print(f"  Eval dir    : {eval_save_dir}")
    print(f"  Predictions : {predict_save_dir}")
    print()
    if map50 is not None:
        print(f"  mAP@0.50    : {map50:.4f}")
    if map50_95 is not None:
        print(f"  mAP@0.5:0.95: {map50_95:.4f}")
    if precision is not None:
        print(f"  Precision   : {precision:.4f}")
    if recall is not None:
        print(f"  Recall      : {recall:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
