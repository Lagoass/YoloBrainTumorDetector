# IMPLEMENTATION PLAN: YOLOv11s Evaluation & Inference

## OBJECTIVE
Create scripts to rigorously evaluate the trained YOLOv11s model on unseen medical data (the `test` split) and generate visual predictions so the user can inspect the AI's accuracy in detecting brain tumors.

## STEP 1: Formal Evaluation Script (`src/evaluate.py`)
Create a script to generate scientific metrics (mAP, Precision, Recall, Confusion Matrix) using the test dataset.
### 1.1 Implementation Details
- **Library**: `ultralytics`
- **Model Path**: Load the best weights from the previous training run (e.g., `runs/brain_tumor/yolo11s_run1/weights/best.pt`). *Hint: Allow the user to pass the path as an argument, or auto-detect the latest run.*
- **Method**: Call `model.val()` with `data=DATA_YAML` and specifically force `split="test"` so it doesn't just re-evaluate the validation set.
- **Output**: The YOLO library will automatically generate evaluation metrics and save them inside a new folder (e.g., `runs/detect/val`). Ensure the script prints where these results were saved.

## STEP 2: Visual Inference Script (`src/predict.py`)
Create a script that picks a few random images from the `test` split and visually draws the bounding boxes over the tumors.
### 2.1 Implementation Details
- **Library**: `ultralytics`, `os`, `random`, `glob`
- **Model Path**: Load the same `best.pt` weights.
- **Input**: Select 10 random images from `data/dataset/images/test/`.
- **Method**: Call `model.predict(source=images, save=True, conf=0.25)`.
- **Output**: YOLO will automatically save the annotated `.jpg` files in a `runs/detect/predict` folder.

## EXECUTION INSTRUCTION FOR CLAUDE CODE
Please read this plan and implement both `src/evaluate.py` and `src/predict.py` exactly as described. The code must be clean, use relative/absolute paths resolving from the project root (`Path(__file__).resolve().parent.parent`), and include proper error handling in case the `best.pt` file does not exist yet.
