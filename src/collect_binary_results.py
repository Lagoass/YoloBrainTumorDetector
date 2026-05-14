"""
One-shot utility: collects ratio ablation results into
runs/analize_models/binary_ratio_comparison/run_all/
"""

import shutil
from pathlib import Path

RATIOS = [10, 20, 30, 40]
TUMORS = ["glioma", "meningioma", "pituitary"]

BASE_RUNS = Path("runs/binary_brain_tumor")
OUTPUT_DIR = Path("runs/analize_models/binary_ratio_comparison/run_all")

# collected[tumor] = list of ratio strings; skipped = flat list of warnings
collected = {tumor: [] for tumor in TUMORS}
skipped = []

for ratio in RATIOS:
    for tumor in TUMORS:
        search_dir = BASE_RUNS / f"{ratio}pct" / tumor
        if not search_dir.exists():
            skipped.append(f"{ratio}pct/{tumor}: run directory not found ({search_dir})")
            continue

        run_dirs = [d for d in search_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            skipped.append(f"{ratio}pct/{tumor}: no run folders found under {search_dir}")
            continue

        latest = max(run_dirs, key=lambda d: d.stat().st_mtime)

        results_src = latest / "results.csv"
        cm_src = latest / "eval_test" / "confusion_matrix.png"

        missing = []
        if not results_src.exists():
            missing.append("results.csv")
        if not cm_src.exists():
            missing.append("eval_test/confusion_matrix.png")

        if missing:
            skipped.append(f"{ratio}pct/{tumor} ({latest.name}): missing {', '.join(missing)}")
            continue

        tumor_dir = OUTPUT_DIR / tumor
        tumor_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(results_src, tumor_dir / f"results_{ratio}pct_{tumor}.csv")
        shutil.copy2(cm_src, tumor_dir / f"confusion_matrix_{ratio}pct_{tumor}.png")

        collected[tumor].append(f"{ratio}pct")

print(f"\n=== collect_binary_results summary ===")
print(f"Output: {OUTPUT_DIR.resolve()}\n")

total_collected = sum(len(v) for v in collected.values())
if total_collected:
    print(f"Collected ({total_collected}):")
    for tumor in TUMORS:
        if collected[tumor]:
            ratios_str = ", ".join(collected[tumor])
            print(f"  {tumor}: {ratios_str}")

if skipped:
    print(f"\nWarnings ({len(skipped)}):")
    for entry in skipped:
        print(f"  [WARN] {entry}")

print(f"\nTotal: {total_collected} collected, {len(skipped)} warnings.")
