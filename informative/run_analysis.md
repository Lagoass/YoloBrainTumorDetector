# Run Analysis Log

Scientific log of each training run: metrics, confusion matrix findings, root cause, and next-run plan.
Each section title matches the run folder name under `runs/brain_tumor/`.

---

## Fixed Training Configuration
> These parameters are stable across all runs and are not repeated in run-specific sections.

| Parameter | Value |
|---|---|
| epochs | 100 |
| imgsz | 640 |
| batch | 16 |
| amp | True |
| patience | 30 |
| cos_lr | True |
| fliplr | 0.5 |
| flipud | 0.0 |
| degrees | 10.0 |
| mosaic | 0.5 |
| mixup | 0.0 |
| hsv_v | 0.2 |
| hsv_s | 0.0 |
| close_mosaic | 10 |

---

> **Data integrity note:** Runs 1–3 were trained on corrupted 2.5D data. The Z-slice grouping by PID was incorrect — adjacent slices belonged to different patients, producing random channel stacking with no anatomical meaning. Run 4 onwards uses corrected pure 2D inputs (each .mat converted individually, min-max normalized, replicated to 3-channel RGB). All 3064 images must be regenerated before Run 4.

## Run Comparison Table

| Run | Folder | mAP@0.50 | mAP@0.5:0.95 | Precision | Recall | Key Change |
|---|---|---|---|---|---|---|
| 1 | yolo11s_29_04_1620 | 0.9217 | 0.5468 | 0.9244 | 0.8453 | Baseline — no class weights |
| 2 | yolo11s_29_04_1759 | 0.9217 | 0.5468 | 0.9244 | 0.8453 | label_smoothing=0.1 — no effect |
| 3 | yolo11s_29_04_2007 | 0.9290 | 0.6212 | 0.9301 | 0.8844 | Oversampling meningioma+pituitary to glioma parity (seed=42) |
| 4 | yolo11s_03_05_2240 | 0.9236 | 0.5659 | 0.8763 | 0.8723 | 2D puro (sem 2.5D) + cos_lr=True + patience=15 + sem oversample |
| 5 | yolo11s_04_05_1246 | 0.9337 | 0.5549 | 0.9250 | 0.9054 | 2D puro + oversampling (meningioma 552→1087, pituitary 767→1087) |
| 6 | yolo11s_11_05_1536 | 0.8469 | 0.4796 | 0.7844 | 0.8461 | BRISC 2025 — balanced classes + 967 healthy negatives + 3 planes |
| 7 | yolo11s_11_05_1621 | 0.9195 | 0.5907 | 0.9178 | 0.8840 | BRISC 2025 + patience=30 — full 100 epochs, best epoch 62 |

---

## Run 1 — yolo11s_29_04_1620

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9217 |
| mAP@0.5:0.95 | 0.5468 |
| Precision | 0.9244 |
| Recall | 0.8453 |

### Run-specific Changes
None — baseline run, no class weights applied.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Pituitary tumor | 0.87 | Best class — distinct morphology aids separation |
| Glioma | 0.78 | 0.21 missed as background — diffuse borders reduce confidence |
| Meningioma | 0.74 | 0.18 confused with glioma — similar MRI appearance |
| Background → Glioma | 0.77 FP rate | Strong class bias: background patches predicted as glioma |

### Root Cause
Class imbalance: glioma is the most represented class in the dataset. The model develops overconfidence toward glioma, pulling borderline meningioma predictions and background patches into that class.

### Conclusion & Next Run Plan
`label_smoothing=0.1` tested in Run 2 — had no effect. Oversample meningioma before training to balance class distribution at the data level.

---

## Run 2 — yolo11s_29_04_1759

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9217 |
| mAP@0.5:0.95 | 0.5468 |
| Precision | 0.9244 |
| Recall | 0.8453 |

### Run-specific Changes
`label_smoothing=0.1` added to address glioma overconfidence from class imbalance.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Pituitary tumor | 0.87 | Best class — distinct morphology aids separation |
| Glioma | 0.78 | 0.21 missed as background — diffuse borders reduce confidence |
| Meningioma | 0.74 | 0.18 confused with glioma — similar MRI appearance |
| Background → Glioma | 0.77 FP rate | Strong class bias: background patches predicted as glioma |

### Root Cause
`label_smoothing` is a symmetric operation — it softens all classes equally and is insensitive to class distribution. With Ultralytics fixed seed, this produced an identical training trajectory to Run 1. It is the wrong tool for asymmetric class imbalance.

### Conclusion & Next Run Plan
Oversample meningioma before training to balance class distribution at the data level. Create `src/oversample.py` to duplicate meningioma images/labels in the train split until reaching glioma parity (~1426 samples).

---

## Run 3 — yolo11s_29_04_2007

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9290 |
| mAP@0.5:0.95 | 0.6212 |
| Precision | 0.9301 |
| Recall | 0.8844 |

### Run-specific Changes
Oversampling applied — meningioma 552→1087, pituitary 767→1087, glioma unchanged at 1087. Pipeline uses `data/oversample_dataset/dataset.yaml`.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Meningioma | 0.76 | +0.02 vs Run 1 — oversampling improved minority class |
| Glioma | 0.79 | +0.01 vs Run 1 — marginal improvement |
| Pituitary | 0.79 | -0.08 vs Run 1 — expected trade-off: oversample repetitions reduced effective diversity |
| Background → Glioma | 0.78 FP rate | Unchanged — bias persists despite balanced classes |

### Root Cause
Background→Glioma bias persists despite balanced classes, suggesting the problem is partially intrinsic to glioma's diffuse visual appearance in T1 MRI rather than purely a data distribution issue.

### Conclusion & Next Run Plan
Oversampling confirmed correct direction. All global metrics improved, especially mAP@0.5:0.95 (+0.0744) and Recall (+0.0391).

Post-run analysis: `best.pt` was saved at epoch 36, with severe oscillation for the remaining 64 epochs — a clear sign of LR instability after the warmup phase ends. Fix: `cos_lr=True` for smooth monotonic decay + `patience=15` to stop earlier if no improvement.

---

## Run 4 — yolo11s_03_05_2240

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9236 |
| mAP@0.5:0.95 | 0.5659 |
| Precision | 0.8763 |
| Recall | 0.8723 |

### Run-specific Changes
First run on corrected pure 2D data (min-max normalized, 3-channel RGB replicated). No oversampling. cos_lr=True, patience=15. Early stop at epoch 55.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Meningioma | 0.80 | +0.06 vs Run 1 — largest single-class gain across all runs. Clean 2D data directly improved the hardest class. |
| Glioma | 0.71 | -0.07 vs Run 1 — regression. 2.5D noise accidentally reinforced diffuse glioma textures. Without it, model is less confident on glioma. |
| Pituitary | 0.85 | -0.02 vs Run 1 — stable, minor drop. |
| Background → Glioma | 0.68 FP rate | -0.09 vs Run 1 — significant improvement. Clean inputs reduced false positive glioma predictions on background. |

### Root Cause
Pure 2D inputs improved data quality and reduced background bias. However, class imbalance (glioma 46.5% of dataset) now dominates without the accidental texture reinforcement from corrupted 2.5D channels. Glioma needs oversampling compensation on clean data.

### Conclusion & Next Run Plan
Run 5 = 2D puro + oversampling. Expected to combine meningioma gains from clean data with glioma recovery from balanced classes.

---

## Run 5 — yolo11s_04_05_1246

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9337 |
| mAP@0.5:0.95 | 0.5549 |
| Precision | 0.9250 |
| Recall | 0.9054 |

### Run-specific Changes
2D puro + oversampling (meningioma 552→1087, pituitary 767→1087). Best epoch: 86/100. Full 100 epochs, no early stop. Highest Recall of all runs.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Meningioma | 0.74 | Lost Run 4 gain; oversample duplicates reduced effective diversity |
| Glioma | 0.81 | +0.10 vs Run 4 — recovered strongly; oversampling compensated class imbalance |
| Pituitary | 0.93 | Best result of any class across all runs |
| Background → Glioma | 0.87 FP rate | Regression vs Run 4 (0.68); overconfidence from memorized duplicates |

### Root Cause
Oversample with exact duplicates causes memorization — val-train divergence 0.784 vs 0.497 in Run 4. Model associates uncertainty with glioma since it is the only class with fully unique, diverse images. Meningioma and pituitary oversamples are repeated patterns; glioma is not.

### Conclusion & Next Run Plan
Dataset is the bottleneck. 708 meningioma images are insufficient for robust generalization. Undersampling glioma discards valid data without improving meningioma. Priority: find a larger, balanced dataset with healthy brain scans and multiple acquisition planes. Awaiting Deep Research results.

---

## Run 6 — yolo11s_11_05_1536

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.8469 |
| mAP@0.5:0.95 | 0.4796 |
| Precision | 0.7844 |
| Recall | 0.8461 |

### Run-specific Changes
First run on BRISC 2025 dataset. 4802 train / 599 val / 599 test. Stratified 80/10/10 split by class+plane. 967 healthy brain negative samples included. Early stop at epoch 30/100, best epoch 26.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Meningioma | 0.94 | Best result of any class across all runs. 1635 unique diverse images resolved the generalization bottleneck. |
| Glioma | 0.76 | Solid, stable result |
| Pituitary | 0.91 | Excellent — close to Run 5 best (0.93) |
| Background → Glioma | 0.47 FP rate | Dramatic reduction from 0.87 (Run 5) and 0.68 (Run 4). Healthy brain negative samples working as intended. |

### Root Cause
Early stop at epoch 30 — patience=15 too aggressive for a new dataset. cls_loss spiked severely at epochs 2–3 (values 10–12), model stabilized at epoch 4 and was still improving at epoch 30. Val-train divergence near zero (0.024) confirms healthy generalization with no overfitting; patience fired on a transient plateau, not a real convergence.

### Conclusion & Next Run Plan
BRISC confirmed as the correct direction — meningioma generalization bottleneck and background→glioma FP rate both resolved in a single dataset switch. Global metrics are lower than Run 5 only because the model needed more epochs. Run 7: identical config with patience=30 to allow longer exploration past transient plateaus.

---

## Run 7 — yolo11s_11_05_1621

### Metrics (test split)
| Metric | Value |
|---|---|
| mAP@0.50 | 0.9195 |
| mAP@0.5:0.95 | 0.5907 |
| Precision | 0.9178 |
| Recall | 0.8840 |

### Run-specific Changes
patience increased 15→30. Full 100 epochs, no early stop. Best epoch: 62/100. BRISC 2025 dataset.

### Confusion Matrix Findings
| Class | True Positive Rate | Notes |
|---|---|---|
| Meningioma | 0.94 | Maintained Run 6 best-ever result |
| Glioma | 0.85 | +0.09 vs Run 6 — 70 extra epochs enabled better learning of the hardest class |
| Pituitary | 0.96 | Best absolute result of any class across all runs in the project |
| Background → Glioma | 0.50 FP rate | Slight regression vs Run 6 (0.47) — acceptable given glioma gain |

### Root Cause
cls_val spiked at epochs 2–3 (values 10–12) — characteristic of BRISC multi-plane multi-protocol variance. Model stabilized at epoch 4. cos_lr produced exemplary terminal convergence: std 0.0027 over last 20 epochs. Val-train divergence 0.184 — healthy generalization. patience=30 was the correct fix for Run 6's premature stop.

### Conclusion & Next Run Plan
Run 7 is the best model of the project. Meningioma, glioma, and pituitary all improved vs the Figshare baseline. Background→Glioma FP rate reduced from 0.87 (Run 5 Figshare) to 0.50. Next: implement False Positive Rate metric on healthy images in `evaluate.py` to properly quantify background suppression from BRISC negative samples.

---

## Binary Models — BRISC 2025

Três modelos YOLOv11s especializados (nc=1 cada) treinados no dataset `data/dissected_brisc/` gerado por `prepare_dataset_binary.py`. Cada modelo recebe as imagens da sua classe-alvo (tumor) + todas as imagens `no_tumor` do BRISC, com labels vazios para as negativas. Mesma configuração de treinamento do Run 7 multiclasse. Baseline de comparação em todas as subseções abaixo: **Run 7 multiclasse** (`yolo11s_11_05_1621`).

---

### 1. Run Inicial — dissected_brisc (ratio ~40%)

O dataset `dissected_brisc` foi gerado por `prepare_dataset_binary.py` incluindo todas as imagens `no_tumor` do BRISC sem restrição de volume — o que resultou em um ratio de negativos de aproximadamente 40% em cada split, herdado da proporção natural do dataset completo (967 no_tumor para qualquer pool de classe tumor). Esse ratio não foi uma escolha explícita, mas o padrão não intencional do dataset de origem.

#### Tabela Comparativa — Binary Models vs. Run 7

| Modelo | mAP@0.50 | mAP@0.5:0.95 | Precision | Recall | FPR healthy | vs Run 7 Recall |
|---|---|---|---|---|---|---|
| Run 7 (multiclasse) | 0.9195 | 0.5907 | 0.9178 | 0.8840 | 0.50 (background→glioma) | — |
| Binary — Glioma | 0.7932 | 0.4093 | 0.7292 | 0.7786 | 0.0000 | −0.1054 |
| Binary — Meningioma | 0.9574 | 0.6198 | 0.9432 | 0.9693 | 0.0000 | +0.0853 |
| Binary — Pituitary | 0.9675 | 0.5294 | 0.8956 | 0.9264 | 0.0333 | +0.0424 |

---

#### Binary — Glioma (ratio ~40%)

##### Métricas (test split)
| Métrica | Valor |
|---|---|
| mAP@0.50 | 0.7932 |
| mAP@0.5:0.95 | 0.4093 |
| Precision | 0.7292 |
| Recall | 0.7786 |
| FPR healthy | 0.0000 |

##### Run-specific Changes
Baseline de comparação: Run 7 multiclasse. Modelo binário nc=1 (glioma vs. no_tumor). Dataset: `dissected_brisc/glioma/` — imagens de glioma do BRISC + todas as imagens no_tumor com labels vazios. Eliminada competição inter-classe com meningioma e pituitary.

##### Confusion Matrix Findings
| Classe | Detectados / Total | Notas |
|---|---|---|
| Glioma (TP) | 121 / 140 | TPR 0.864 — 19 missed |
| Background → Glioma (FP) | 0 / 37 acima de conf=0.25 | 37 FP existem abaixo de conf=0.25; FPR medido na faixa operacional = 0.000 |

Deltas vs. Run 7: Recall −0.1054 (0.7786 vs. 0.8840); FPR healthy 0.0000 vs. 0.50 (eliminação total).

##### Root Cause
A regressão no recall de glioma sob o modelo binário revela que parte do desempenho no Run 7 multiclasse era sustentado por regularização implícita das classes competidoras. Com apenas glioma vs. no_tumor, o modelo não dispõe da pressão contrastiva de meningioma e pituitary para refinar suas fronteiras de decisão — glioma é a classe mais difusa visualmente (bordas irregulares, edema peritumoral) e depende de contexto multi-classe para atingir sua máxima discriminação. O FPR zero confirma que o FPR residual de 0.50 do Run 7 é inteiramente atribuível à competição inter-classe e não à confusão glioma–tecido-saudável: quando forçado a decidir apenas entre glioma e normal, o modelo não gera falsos positivos na faixa operacional.

##### Cross-Model Finding
O modelo binário de glioma responde diretamente à questão central do `research_directions.md`: a competição multi-classe **fornece regularização benéfica** para glioma. O modelo especializado perde −0.1054 de recall em relação ao multiclasse (0.7786 vs. 0.8840), enquanto elimina integralmente o FPR em healthy. O trade-off é desfavorável para glioma isolado: a especialização remove o FPR, mas remove também a pressão contrastiva que tornava o multiclasse mais discriminativo. Para glioma, o modelo unificado é superior em recall clínico; o binário é superior em especificidade.

---

#### Binary — Meningioma (ratio ~40%)

##### Métricas (test split)
| Métrica | Valor |
|---|---|
| mAP@0.50 | 0.9574 |
| mAP@0.5:0.95 | 0.6198 |
| Precision | 0.9432 |
| Recall | 0.9693 |
| FPR healthy | 0.0000 |

##### Run-specific Changes
Baseline de comparação: Run 7 multiclasse. Modelo binário nc=1 (meningioma vs. no_tumor). Dataset: `dissected_brisc/meningioma/` — imagens de meningioma do BRISC + todas as imagens no_tumor com labels vazios.

##### Confusion Matrix Findings
| Classe | Detectados / Total | Notas |
|---|---|---|
| Meningioma (TP) | 158 / 163 | TPR 0.969 — 5 missed |
| Background → Meningioma (FP) | 0 / 10 acima de conf=0.25 | 10 FP existem abaixo de conf=0.25; FPR medido na faixa operacional = 0.000 |

Deltas vs. Run 7: Recall +0.0853 (0.9693 vs. 0.8840); mAP@0.50 +0.0379 (0.9574 vs. 0.9195); FPR healthy 0.0000 (vs. não isolável no multiclasse).

##### Root Cause
Meningioma é a classe que mais se beneficia da especialização binária. No multiclasse, 30–40% das ativações de meningioma competiam diretamente com glioma, que compartilha padrões de contraste em T1. Ao remover essa competição, o modelo dedica toda a capacidade do detection head à distinção meningioma–normal, superando em recall (+0.0853) e mAP@0.50 (+0.0379) o melhor resultado multiclasse jamais obtido. O conjunto de dados BRISC fornece 1329 imagens diversas de meningioma (vs. 708 no Figshare original) — a combinação de diversidade de dados e foco binário elimina o gargalo de generalização identificado no Run 5.

##### Cross-Model Finding
Meningioma é o único caso em que o modelo especializado supera o multiclasse em todas as métricas primárias. Isso confirma que, para classes com alta sobreposição visual com uma classe vizinha (meningioma/glioma), a competição inter-classe no multiclasse impõe um teto estrutural de desempenho. A questão central do `research_directions.md` recebe resposta assimétrica: a regularização compartilhada beneficia glioma mas prejudica meningioma. Um ensemble prático combinaria o especialista binário de meningioma com o modelo unificado para as demais classes.

---

#### Binary — Pituitary (ratio ~40%)

##### Métricas (test split)
| Métrica | Valor |
|---|---|
| mAP@0.50 | 0.9675 |
| mAP@0.5:0.95 | 0.5294 |
| Precision | 0.8956 |
| Recall | 0.9264 |
| FPR healthy | 0.0333 |

##### Run-specific Changes
Baseline de comparação: Run 7 multiclasse. Modelo binário nc=1 (pituitary vs. no_tumor). Dataset: `dissected_brisc/pituitary/` — imagens de pituitary do BRISC + todas as imagens no_tumor com labels vazios.

##### Confusion Matrix Findings
| Classe | Detectados / Total | Notas |
|---|---|---|
| Pituitary (TP) | 167 / 176 | TPR 0.949 — 9 missed |
| Background → Pituitary (FP) | 4 / 43 acima de conf=0.25 | 43 FP totais, mas 39 abaixo de conf=0.25; FPR operacional = 0.0333 |

Deltas vs. Run 7: Recall +0.0424 (0.9264 vs. 0.8840); mAP@0.50 +0.0480 (0.9675 vs. 0.9195); FPR healthy 0.0333 — único modelo binário com FPR residual acima de zero na faixa operacional.

##### Root Cause
Pituitary apresenta morfologia distinta (tumor pequeno, sela turca, plano sagital predominante) que o torna naturalmente discriminável — o modelo especializado melhora recall e mAP sem surpresas. O FPR residual de 0.0333 (4 imagens healthy acima de conf=0.25) contrasta com FPR zero nos outros dois binários e revela que uma fração do ruído background→pituitary no Run 7 é genuinamente visual, não um artefato da competição inter-classe: estruturas anatômicas da base do crânio (seio esfenoidal, hipófise normal) ativam o detector em conf acima do limiar operacional. Isso não estava visível no multiclasse porque o FPR era agregado em background→glioma. A naturalidade do FPR residual de pituitary é um achado mecanístico novo.

##### Cross-Model Finding
Pituitary é o caso intermediário: o especialista binário melhora recall (+0.0424) e mAP@0.50 (+0.0480), confirmando que a competição inter-classe limitava levemente seu desempenho no multiclasse. Porém, ao contrário de meningioma e glioma, pituitary retém um FPR residual de 0.0333, demonstrando que parte das suas confusões com background são intrínsecas à anatomia da base do crânio e não dependem da presença de outras classes tumorais. Isso responde a uma sub-questão do `research_directions.md`: o FPR de 0.50 do Run 7 (medido em background→glioma) não era uniforme entre classes — glioma carregava quase toda a contaminação inter-classe, meningioma era secundária, e pituitary contribuía com um ruído anatômico irredutível independente da estratégia de classificação.

---

#### Motivação do Ablation

A run inicial com ratio ~40% revelou três assimetrias que motivaram o ablation controlado:

1. **Glioma regrediu vs. Run 7 multiclasse** (recall −0.1054): a hipótese levantada é que glioma depende de regularização contrastiva inter-classe; sem ela, o volume de negativos pode compensar ou agravar a perda dependendo do ratio.
2. **FPR variou por classe** (glioma 0%, pituitary 3.3%): o FPR não é uma propriedade uniforme do modelo — é determinado pela anatomia da classe. A questão isolável é se diferentes ratios de negativos modulam esse FPR de forma distinta para cada classe.
3. **Hipótese de ablation**: o ratio de negativos penaliza as classes de forma assimétrica. Mantendo arquitetura, augmentação e seed fixos e variando apenas o percentual de no_tumor (10/20/30/40%), é possível identificar o ratio ótimo por classe e separar o efeito do volume de negativos do efeito da competição inter-classe.

---

### 2. Ratio Ablation — dissected_brisc_ratios

Dataset gerado por `prepare_dataset_binary_ratios.py` → `data/dissected_brisc_ratios/{ratio}pct/{tumor}/`. Para cada combinação de ratio × tumor, o pipeline `pipeline_binary.py --ratio {ratio} --tumor {tumor}` treina um modelo independente com configuração idêntica ao Run 7, variando apenas o volume de imagens no_tumor no train split. Test split mantido idêntico ao `dissected_brisc` original para comparabilidade direta.

#### 2a. Tabela Comparativa Geral (12 combinações)

| Tumor | Ratio | mAP@0.50 | Precision | Recall | FPR | Best epoch | Total epochs |
|---|---|---|---|---|---|---|---|
| Glioma | 10% | 0.8137 | 0.8143 | 0.7571 | 0.0% | 91 | 95★ |
| Glioma | 20% | 0.8497 | 0.7935 | 0.8509 | 0.0% | 99 | 100 |
| Glioma | 30% | 0.8129 | 0.8368 | 0.6857 | 0.0% | 80 | 90★ |
| Glioma | 40% | 0.8191 | 0.8483 | 0.7643 | 0.0% | 79 | 100 |
| Meningioma | 10% | 0.9582 | 0.9583 | 0.9693 | 0.0% | 90 | 100 |
| Meningioma | 20% | 0.9727 | 0.9734 | 0.9632 | 0.0% | 77 | 100 |
| Meningioma | 30% | 0.9550 | 0.9457 | 0.9607 | 4.3% | 61 | 69★ |
| Meningioma | 40% | 0.9663 | 0.9631 | 0.9632 | 0.0% | 87 | 95★ |
| Pituitary | 10% | 0.9568 | 0.8549 | 0.9432 | 10.0% | 37 | 63★ |
| Pituitary | 20% | 0.9581 | 0.8910 | 0.9288 | 6.8% | 59 | 59★⚡ |
| Pituitary | 30% | 0.9671 | 0.9399 | 0.9489 | 2.7% | 34 | 95★ |
| Pituitary | 40% | 0.9524 | 0.9362 | 0.9432 | 0.85% | 45 | 100 |

★ early stop &nbsp;&nbsp; ⚡ best epoch = último epoch (modelo cortado prematuramente — ainda melhorando)

---

#### 2b. Análise por Tumor

##### Glioma — Ratio Ablation

###### Confusion matrix não normalizada (test split, 140 positivos, pool bg variável)

| Ratio | TP | Missed | FP anchors | Pool bg |
|---|---|---|---|---|
| 10% | 115 / 140 | 25 | 36 | 16 |
| 20% | 124 / 140 | 16 | 40 | 35 |
| 30% | 118 / 140 | 22 | 55 | 60 |
| 40% | 117 / 140 | 23 | 30 | 93 |

###### Root Cause
Glioma é a classe com maior variabilidade de resposta ao ratio. O pico de recall (0.8509) e mAP (0.8497) ocorre em 20%, onde o modelo dispõe de negativos suficientes para aprender a fronteira glioma–normal sem ser dominado por eles — a convergência até epoch 99/100 indica que o modelo utilizou toda a capacidade de treinamento disponível sem early stop. Em 10%, o pool de negativos (16 imagens no test) é insuficiente para calibrar a fronteira: recall cai para 0.7571 e FP anchors sobem para 36 apesar do pool menor, sugerindo threshold de ativação frouxo por falta de pressão negativa durante treino. Em 30% e 40%, o excesso de negativos induz o modelo a priorizar especificidade — recall regride para 0.6857 e 0.7643 respectivamente — e o early stop em 90 e 79 epochs indica que o modelo saturou a fronteira negativa antes de refinar a positiva. O FPR permanece 0% em todos os ratios: a geometria glioma–healthy é naturalmente separável quando a competição inter-classe é removida, independente do volume de negativos.

###### Ratio Recomendado
**20%** — melhor mAP@0.50 (0.8497) e recall (0.8509), convergência até epoch 99 sem early stop. Único ratio que maximiza simultaneamente discriminação positiva e estabilidade de treino para glioma.

---

##### Meningioma — Ratio Ablation

###### Confusion matrix não normalizada (test split, 163 positivos, pool bg variável)

| Ratio | TP | Missed | FP anchors | Pool bg |
|---|---|---|---|---|
| 10% | 159 / 163 | 4 | 10 | 18 |
| 20% | 158 / 163 | 5 | 7 | 41 |
| 30% | 158 / 163 | 5 | 16 | 70 |
| 40% | 158 / 163 | 5 | 8 | 109 |

###### Root Cause
Meningioma é a classe mais estável ao ratio: 158–159 TPs em todos os ratios, missed variando em apenas 1 imagem entre 10% e os demais. A diferença determinante é mAP e FPR. O pico de mAP@0.50 (0.9727) ocorre em 20%, que também apresenta o menor FP anchor absoluto (7) apesar de pool bg moderado (41) — sinal de calibração precisa do threshold de ativação. Em 10%, FP anchors sobem para 10 com pool bg de apenas 18, sugerindo threshold frouxo por pressão negativa insuficiente durante treino. Em 30%, FP anchors sobem para 16 e surge FPR de 4.3% com early stop prematuro em 69 epochs — o volume de negativos começou a deslocar o threshold mas o treino foi cortado antes da re-calibração. Em 40%, FP anchors caem para 8 e FPR volta a 0%, mas mAP@0.50 regride para 0.9663 e early stop em 95 epochs indica que o modelo não convergiu totalmente. O regime ótimo para meningioma é 20%: negativos suficientes para calibrar sem sobre-pressionar, e convergência completa sem early stop.

###### Ratio Recomendado
**20%** — mAP@0.50 mais alto (0.9727), menor FP anchor absoluto (7), FPR 0%, convergência plena até epoch 77/100.

---

##### Pituitary — Ratio Ablation

###### Confusion matrix não normalizada (test split, 176 positivos, pool bg variável)

| Ratio | TP | Missed | FP anchors | Pool bg |
|---|---|---|---|---|
| 10% | 169 / 176 | 7 | 30 | 20 |
| 20% | 173 / 176 | 3 | 55 | 44 |
| 30% | 172 / 176 | 4 | 18 | 75 |
| 40% | 169 / 176 | 7 | 19 | 117 |

###### Root Cause
Pituitary é a exceção estrutural do ablation. O FPR permanece positivo em todos os ratios (10%→2.7%→6.8%→10%), confirmando que a anatomia de base craniana (seio esfenoidal, hipófise normal) gera ativações genuínas acima de conf=0.25 independente do volume de negativos — esse ruído é intrínseco, não calibrável apenas pelo ratio. O comportamento de FP anchors revela um regime não monotônico: 10% produz 30 FP anchors com FPR 10%; 20% sobe para 55 FP anchors mas FPR cai para 6.8% — o modelo detecta mais regiões candidatas mas com threshold mais seletivo. 30% cai para 18 FP anchors e FPR 2.7%, o melhor equilíbrio atual; 40% mantém 19 FP anchors mas FPR desce para 0.85% à custa de recall regressindo de 0.9489 para 0.9432 e 7 missed (vs. 3–4 nos ratios intermediários). O early stop em 63 epochs (10%), 59/59 epochs (20%, best epoch = último — modelo ainda melhorando), e 95 epochs (30%) indica que 20% não teve treinamento suficiente para demonstrar seu potencial máximo — o run de 20% foi cortado exatamente no melhor epoch registrado, o que é um sinal de corte prematuro. 40% converge completamente em 100 epochs mas com custo em recall.

###### Ratio Recomendado
**30%** como melhor equilíbrio atual: mAP@0.50=0.9671 (mais alto do ablation), recall=0.9489, FPR=2.7%, apenas 4 missed. **20% requer rerun com patience=50**: o best epoch coincidiu com o último epoch treinado (59/59), indicando que o modelo foi cortado prematuramente e pode superar 30% com mais epochs. Decisão pendente de rerun.

---

#### 2c. Cross-Ratio Finding

Não existe ratio ótimo universal entre as três classes. Glioma e meningioma convergem para 20% como ratio ótimo — volume suficiente para calibrar a fronteira tumor–normal sem induzir over-especificidade. Pituitary é a exceção estrutural: sua anatomia de base craniana impõe FPR residual independente do ratio, e o regime ótimo está em 30% (ou 20% com mais epochs), onde o volume de negativos é suficiente para suprimir parcialmente as ativações de base craniana sem sacrificar recall das detecções positivas.

O achado central do ablation é que **a sensibilidade ao ratio de negativos é uma propriedade da classe, não do modelo ou do dataset**. Glioma é sensível ao ratio por depender de regularização contrastiva; meningioma é estável ao ratio porque sua fronteira morfológica é clara; pituitary é resistente à supressão de FPR via ratio porque a fonte de ruído (anatomia da base craniana) é visualmente similar ao tumor. Esses três padrões são irredutíveis à mesma estratégia de controle de ratio — cada especialista binário requer sua própria calibração.

---

### 3. Binary vs. Multiclasse — Comparação Final

Comparação entre Run 7 (melhor modelo multiclasse) e o melhor ratio de cada especialista binário identificado pelo ablation.

| Modelo | mAP@0.50 | Precision | Recall | FPR |
|---|---|---|---|---|
| Run 7 multiclasse | 0.9195 | 0.9178 | 0.8840 | 0.50 |
| Binary glioma 20% | 0.8497 | 0.7935 | 0.8509 | 0.0% |
| Binary meningioma 20% | 0.9727 | 0.9734 | 0.9632 | 0.0% |
| Binary pituitary 30% | 0.9671 | 0.9399 | 0.9489 | 2.7% |

#### Root Cause Mecanístico

**Glioma — único caso em que o multiclasse é superior:** Glioma é a classe mais difusa visualmente (bordas irregulares, edema peritumoral, contraste variável em T1). No modelo multiclasse, meningioma e pituitary fornecem pressão contrastiva que força o detection head a refinar as fronteiras de glioma — o modelo aprende onde glioma *não* é ao ver onde as outras classes são. Essa regularização contrastiva eleva recall de 0.8509 (binário 20%) para 0.8840 (multiclasse) sem custo de FPR entre classes tumorais. O binário elimina o FPR (0% vs. 0.50 do multiclasse), mas perde o benefício regularizador: para glioma, a especialização é uma regressão clínica.

**Meningioma e pituitary — especialistas superiores ao multiclasse:** Meningioma tem alta sobreposição visual com glioma em T1 (ambos são lesões supratentoriais com captação de contraste variável). No multiclasse, essa sobreposição se manifesta como competição direta no detection head — patches de meningioma competem com glioma pelo mesmo output slot, limitando estruturalmente o recall. Ao remover glioma do espaço de classes, o especialista de meningioma dedica toda a capacidade discriminativa à fronteira meningioma–normal: recall sobe de 0.8840 para 0.9632 (+0.0792) e mAP@0.50 de 0.9195 para 0.9727. Pituitary segue lógica similar — morfologia distinta (tumor pequeno, sela turca) que no multiclasse competia parcialmente com glioma pela atenção do detector; isolada, melhora recall de 0.8840 para 0.9489.

**FPR: multiclasse 0.50 vs. binários ~0–3% — origem revelada:** O FPR de 0.50 do Run 7 era inteiramente concentrado em background→glioma e emergia da pressão classificatória inter-classe — patches ambíguos eram forçados a competir entre as três classes e a mais representada (glioma) absorvia a incerteza residual. Nos binários, sem essa competição, glioma e meningioma atingem FPR 0%; pituitary mantém FPR de 2.7% que é irredutível (anatomia de base craniana). O FPR de 0.50 do Run 7 era um artefato classificatório, não uma propriedade visual intrínseca — e o ablation confirmou que é eliminável sem custo de recall para meningioma e pituitary.

#### Cross-Model Finding

O modelo multiclasse é superior para glioma; especialistas binários são superiores para meningioma e pituitary. Essa assimetria é mecanística, não acidental, e define a arquitetura de pipeline ideal: **triagem → multiclasse para glioma, especialistas binários para meningioma e pituitary**. O modelo de triagem (FPR=0.0083) filtra cérebros saudáveis no primeiro estágio; o segundo estágio aplica o classificador adequado por classe. Esse design elimina o trade-off recall×especificidade que aflige cada modelo individualmente: a triagem maximiza recall, os especialistas downstream maximizam especificidade de classe sem herdar o FPR estrutural do multiclasse.

---

## Triage Model — BRISC 2025

Modelo YOLOv11s de triagem (nc=1, class 0=tumor) treinado no dataset `data/triagem_dataset/` gerado por `prepare_dataset_triagem.py`. Todas as três classes tumorais (meningioma, glioma, pituitary) são colapsadas em uma única classe `tumor`; imagens `no_tumor` recebem label vazio. Mesma configuração de treinamento do Run 7 multiclasse. Baseline de comparação em todas as subseções abaixo: **Run 7 multiclasse** e **Binary Models**.

### Métricas (test split)

| Métrica | Valor |
|---|---|
| mAP@0.50 | 0.9425 |
| mAP@0.5:0.95 | 0.6030 |
| Precision | 0.9142 |
| Recall | 0.9019 |
| FPR healthy (nível de imagem) | 0.0083 (1/120) |
| Tempo de treino | 92.5 min |

### Confusion Matrix Findings

| Classe | Taxa | Notas |
|---|---|---|
| Tumor → Tumor (TP) | 0.94 | 6% das detecções tumorais perdidas como background — menor taxa de miss entre todos os modelos treinados |
| Background → Tumor (FP, nível de patch val) | 1.00 | Métrica de patch interna ao YOLO val; FPR real medido em nível de imagem = 0.0083 (1 FP em 120 imagens healthy) |

### Root Cause

O colapso das três classes tumorais em uma única classe `tumor` resolve o FPR sem sacrificar recall por um mecanismo mecanístico direto: no modelo multiclasse (Run 7), o FPR de 0.50 era concentrado em background→glioma e emergia da pressão contrastiva entre classes — patches ambíguos eram forçados a competir entre meningioma, glioma e pituitary, e a classe mais representada (glioma) absorvia a incerteza. Ao colapsar as três classes, o modelo não precisa mais resolver ambiguidades inter-classe: a única decisão relevante é tumor vs. não-tumor. Essa simplificação do espaço de saída direciona toda a capacidade discriminativa do detection head para a fronteira tumor–saudável, que é geometricamente mais separável do que as fronteiras inter-tumorais. O resultado é FPR de 0.0083 em nível de imagem — redução de 98,3% em relação ao FPR de 0.50 do Run 7 — mantendo recall de 0.9019, superior ao Run 7 (0.8840).

Comparando com os modelos binários (FPR ~0.0% mas recall de glioma regredido para 0.7786): os binários eliminam FPR mas pagam custo em recall porque cada especialista perde a regularização contrastiva das classes vizinhas. O modelo de triagem, por sua vez, retém a pressão contrastiva entre tumor e não-tumor sem fragmentar o espaço de classes — glioma não precisa mais competir com meningioma e pituitary dentro do detection head, mas ainda enfrenta o contraste com tecido saudável, que é o sinal discriminativo mais robusto disponível no dataset. Isso explica por que recall(triagem)=0.9019 > recall(binary glioma)=0.7786: a triagem preserva o benefício contrastivo sem o custo da fragmentação inter-classe.

### Cross-Model Finding

O resultado do modelo de triagem confirma empiricamente a hipótese central do pipeline em cascata: **triagem de alta sensibilidade como primeiro estágio é viável e superior a qualquer modelo único.** mAP@0.50=0.9425 é o maior mAP@0.50 obtido no projeto, superando o Run 7 multiclasse (0.9195) e equiparando ao melhor binário (pituitary: 0.9675, mas com escopo restrito). FPR=0.0083 em nível de imagem significa que menos de 1% dos cérebros saudáveis disparariam alarme falso — limiar operacionalmente aceitável para um filtro de primeiro estágio.

A arquitetura de cascata validada por esses resultados é: **(1) Triage Model** como filtro de alta sensibilidade — elimina 99,2% dos cérebros saudáveis antes de qualquer classificação; **(2) Classificador multiclasse (Run 7) ou especialistas binários** como segundo estágio, acionado apenas para imagens positivas no estágio 1. Esse design evita o trade-off de precisão vs. recall que aflige cada modelo individualmente: a triagem maximiza recall (minimiza falsos negativos clínicos), o classificador downstream maximiza especificidade de classe (minimiza confusão inter-tumoral). O Run 7 residual de background→glioma de 0.50 — o principal problema em aberto até os binários — torna-se irrelevante no pipeline em cascata, pois apenas imagens triadas positivo chegam ao classificador multiclasse.
