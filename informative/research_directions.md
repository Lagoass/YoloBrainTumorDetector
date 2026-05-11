# Research Directions

Two complementary training strategies evaluated on the same BRISC 2025 dataset and YOLOv11s backbone.

---

## Multi-Class Unified Model

A single YOLOv11s trained to detect all three tumor types simultaneously — the approach used in Runs 1–7.

- **Shared representation advantage:** The backbone learns features common to all tumor classes (contrast enhancement patterns, mass effect, peritumoral edema) in a single forward pass, which is efficient at inference and mirrors the real clinical workflow where the tumor type is unknown before the scan is analyzed.
- **Inter-class confusion is measurable and interpretable:** The confusion matrix directly exposes which classes compete (glioma vs. background, meningioma vs. glioma), providing a diagnostic signal for dataset and augmentation decisions — as demonstrated across Runs 1–7.
- **Background suppression requires deliberate effort:** No-tumor negative samples (BRISC healthy images) must be explicitly included in training to reduce false positive rate; without them the background→glioma FP rate reached 0.87 (Run 5).

---

## Binary Specialized Models

Three independent YOLOv11s detectors, one per tumor class: **glioma vs. no_tumor**, **meningioma vs. no_tumor**, **pituitary vs. no_tumor**. Each model trains on BRISC images of its specific tumor class plus all no_tumor images from the dataset, using the same architecture and training configuration as the unified model.

- **Focused decision boundary:** Each model only needs to learn one visual signature against healthy tissue, eliminating inter-class competition. Meningioma — the hardest class in the unified setting — gets a model whose entire capacity is devoted to distinguishing it from normal brain, potentially improving recall beyond the 0.94 achieved in Run 7.
- **Naturally balanced training:** The binary formulation pairs each tumor class (≈1400–1600 BRISC images) directly against no_tumor (≈1207 images), producing near-parity without oversampling or duplication artifacts.
- **False Positive Rate is the primary metric:** With only two possible outcomes (tumor detected / not detected), the FPR on healthy images becomes the central evaluation criterion, directly quantifying clinical safety rather than being a secondary diagnostic.

---

## Cross-Approach Inference

Running both approaches on the same BRISC test split and comparing their outputs reveals information neither approach exposes alone.

- **Attributing confusion sources:** If a binary glioma model still generates false positives on healthy images that the unified model also mislabels, the error is rooted in the visual similarity between glioma texture and normal tissue — not in inter-class competition. If the binary model eliminates those FPs, the unified model's errors are an architectural artifact of shared class heads.
- **Recall vs. specificity trade-off per class:** A binary model may improve recall for its target class but lose the implicit regularization that comes from competing classes. Quantifying this trade-off for each of the three tumor types is a concrete, publishable finding.
- **Ensemble potential:** Disagreement between the unified model and the relevant binary specialist on the same image identifies the highest-uncertainty cases — the subset most relevant for human radiologist review in a clinical decision-support system.
- **Central research question:** Does forcing a single model to distinguish three tumor classes against each other provide beneficial shared-feature regularization, or does it impose a structural ceiling on per-class performance that specialized binary models can overcome?

---

## Conclusion

Training both the unified multi-class model and three binary specialists on identical data and architecture isolates the effect of the classification objective itself, independent of dataset or backbone choices — a clean ablation that is rarely reported in the medical imaging literature. The unified model is the clinically practical artifact and the natural continuation of Runs 1–7; the binary models are the scientific instrument that explains *why* the unified model performs the way it does on each class. Together they produce a complete academic contribution: a deployable detector and a mechanistic account of its failure modes.
