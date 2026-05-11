import sys
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = str(ROOT / "data" / "dataset" / "dataset.yaml")
OVERSAMPLE_YAML = str(ROOT / "data" / "oversample_dataset" / "dataset.yaml")
BRISC_YAML = str(ROOT / "data" / "brisc_dataset" / "dataset.yaml")


def train(epochs=100, batch=16, data_yaml=OVERSAMPLE_YAML):
    run_name = f"yolo11s_{time.strftime('%d_%m_%H%M')}"
    print(f"Starting YOLOv11s training | epochs={epochs} batch={batch} name={run_name}")
    model = YOLO("yolo11s.pt")

    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=batch,
            amp=True,
            workers=4,
            project=str(ROOT / "runs" / "brain_tumor"),
            name=run_name,
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
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch > 8:
            print(f"OOM with batch={batch}, retrying with batch=8")
            model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=640,
                batch=8,
                amp=True,
                workers=4,
                project=str(ROOT / "runs" / "brain_tumor"),
                name=run_name,
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
        else:
            raise
    
    return model.trainer.save_dir, model

def export(save_dir):
    best = Path(save_dir) / "weights" / "best.pt"
    print(f"Exporting {best} to TensorRT engine")
    model = YOLO(str(best))
    model.export(format="engine", half=True, imgsz=640, workspace=4)


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    save_dir, _ = train(epochs=epochs)
    if epochs >= 100:
        export(save_dir)
