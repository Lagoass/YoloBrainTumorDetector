import os
import shutil
import random
import h5py
from tqdm import tqdm

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST = remaining ~10%

RAW_DIR = os.path.join("data", "raw")
IMAGES_ALL = os.path.join("data", "dataset", "images", "all")
LABELS_ALL = os.path.join("data", "dataset", "labels", "all")
DATASET_DIR = os.path.join("data", "dataset")


def get_pid(path):
    with h5py.File(path, "r") as f:
        pid = f["cjdata"]["PID"]
        if isinstance(pid, h5py.Dataset):
            try:
                return "".join([chr(c[0]) for c in pid[:]])
            except Exception:
                return str(pid[0, 0])
        return str(pid)


def build_pid_map():
    mat_files = sorted(
        [f for f in os.listdir(RAW_DIR) if f.endswith(".mat") and f != "cvind.mat"],
        key=lambda x: int(x.split(".")[0]),
    )

    pid_to_indices = {}
    errors = []

    for fname in tqdm(mat_files, desc="Lendo PIDs"):
        idx = int(fname.split(".")[0])
        if not os.path.exists(os.path.join(IMAGES_ALL, f"{idx}.jpg")):
            continue
        try:
            pid = get_pid(os.path.join(RAW_DIR, fname))
            pid_to_indices.setdefault(pid, []).append(idx)
        except Exception as e:
            errors.append((fname, str(e)))

    if errors:
        print(f"\nAvisos: {len(errors)} arquivo(s) ignorado(s)")
        for f, e in errors[:5]:
            print(f"  {f}: {e}")

    return pid_to_indices


def split_pids(pid_to_indices):
    pids = list(pid_to_indices.keys())
    random.seed(SEED)
    random.shuffle(pids)

    n = len(pids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    return {
        "train": pids[:n_train],
        "val": pids[n_train : n_train + n_val],
        "test": pids[n_train + n_val :],
    }


def copy_split(splits, pid_to_indices):
    for split in splits:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    totals = {}
    for split, pids in splits.items():
        indices = [idx for pid in pids for idx in pid_to_indices[pid]]
        totals[split] = {"pids": len(pids), "slices": len(indices)}
        for idx in tqdm(indices, desc=f"Copiando {split:5s}"):
            shutil.copy2(
                os.path.join(IMAGES_ALL, f"{idx}.jpg"),
                os.path.join(DATASET_DIR, "images", split, f"{idx}.jpg"),
            )
            shutil.copy2(
                os.path.join(LABELS_ALL, f"{idx}.txt"),
                os.path.join(DATASET_DIR, "labels", split, f"{idx}.txt"),
            )

    return totals


def write_yaml(totals):
    dataset_abs = os.path.abspath(DATASET_DIR)
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")

    lines = [
        f"path: {dataset_abs}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        "",
        "nc: 3",
        "names:",
        "  0: meningioma",
        "  1: glioma",
        "  2: pituitary tumor",
        "",
        "# Split summary (seed=42, PID-stratified, sem data leakage):",
    ]
    for split, info in totals.items():
        lines.append(f"# {split:5s}: {info['pids']} pacientes, {info['slices']} fatias")

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\ndataset.yaml salvo em: {yaml_path}")


def main():
    print("=== Split por PID (sem data leakage) ===\n")

    pid_to_indices = build_pid_map()
    print(f"\nTotal: {len(pid_to_indices)} pacientes únicos\n")

    splits = split_pids(pid_to_indices)
    totals = copy_split(splits, pid_to_indices)
    write_yaml(totals)

    print("\n=== Resultado ===")
    total_slices = 0
    for split, info in totals.items():
        print(f"  {split:5s}: {info['pids']:3d} pacientes | {info['slices']:4d} fatias")
        total_slices += info["slices"]
    print(f"  Total: {total_slices} fatias")


if __name__ == "__main__":
    main()
