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


DATASET_LABELS = {
    "figshare": "figshare (base)",
    "figshare_oversample": "figshare (oversampled)",
    "brisc": "BRISC 2025",
}


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor YOLO full pipeline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use latest best.pt")
    parser.add_argument(
        "--dataset",
        choices=["figshare", "figshare_oversample", "brisc"],
        default="figshare",
        help="Dataset to train and evaluate on (default: figshare)",
    )
    args = parser.parse_args()

    weights_path: Path | None = None
    trainer = None
    data_yaml_used: str | None = None

    # --- Train ---
    if args.skip_train:
        weights_path = find_latest_weights()
        if weights_path is None:
            print("Error: --skip-train specified but no best.pt found under runs/brain_tumor/")
            sys.exit(1)
        print(f"Skipping training. Using weights: {weights_path}")
    else:
        from train import train, DATA_YAML, OVERSAMPLE_YAML, BRISC_YAML
        if args.dataset == "figshare":
            data_yaml_used = DATA_YAML
        elif args.dataset == "figshare_oversample":
            data_yaml_used = OVERSAMPLE_YAML
        else:
            data_yaml_used = BRISC_YAML
        save_dir, trained_model = train(epochs=args.epochs, data_yaml=data_yaml_used)
        trainer = trained_model.trainer
        weights_path = Path(save_dir) / "weights" / "best.pt"
        print(f"\nTraining complete. Weights: {weights_path}")

    # --- Evaluate ---
    from evaluate import evaluate
    if args.dataset == "figshare":
        from train import DATA_YAML as _data_yaml
    elif args.dataset == "figshare_oversample":
        from train import OVERSAMPLE_YAML as _data_yaml
    else:
        from train import BRISC_YAML as _data_yaml
    metrics, eval_save_dir = evaluate(weights_path, data_yaml=_data_yaml)

    # --- Predict ---
    from predict import predict
    if args.dataset == "brisc":
        test_images_dir = ROOT / "data" / "brisc_dataset" / "images" / "test"
    else:
        test_images_dir = ROOT / "data" / "dataset" / "images" / "test"
    results = predict(weights_path, test_images_dir=test_images_dir)
    predict_save_dir = Path(results[0].save_dir) if results else ROOT / "runs" / "brain_tumor" / "predict"

    # --- Summary ---
    map50 = getattr(metrics.box, "map50", None)
    map50_95 = getattr(metrics.box, "map", None)
    precision = getattr(metrics.box, "mp", None)
    recall = getattr(metrics.box, "mr", None)

    # --- Training metadata ---
    best_epoch = total_time = early_stopped = None
    if trainer is not None:
        best_epoch = getattr(trainer, "best_epoch", None)
        total_time = getattr(trainer, "t", None)
        epochs_run = getattr(trainer, "epoch", None)
        max_epochs = getattr(trainer, "epochs", args.epochs)
        if epochs_run is not None:
            early_stopped = (epochs_run + 1) < max_epochs

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Weights     : {weights_path}")
    print(f"  Eval dir    : {eval_save_dir}")
    print(f"  Predictions : {predict_save_dir}")
    print()
    print(f"  Dataset     : {DATASET_LABELS[args.dataset]}")
    if best_epoch is not None:
        print(f"  Best epoch  : {best_epoch + 1}")
    if total_time is not None:
        h, rem = divmod(int(total_time), 3600)
        m, s = divmod(rem, 60)
        print(f"  Train time  : {h:02d}h {m:02d}m {s:02d}s")
    if early_stopped is not None:
        if early_stopped:
            print(f"  Early stop  : yes (stopped at epoch {epochs_run + 1}/{max_epochs})")
        else:
            print(f"  Early stop  : no (full {max_epochs} epochs)")
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
