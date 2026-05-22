# Otimização por Classe em Detecção de Tumor Cerebral com YOLO

Detecção automática de tumores cerebrais em MRI T1CE via
bounding boxes 2D, com ablation do ratio de amostras negativas
e comparação entre modelo multiclasse e especialistas binários.

---

## O que é este projeto

Um sistema de detecção e localização de tumores cerebrais em
exames de ressonância magnética T1 com contraste de gadolínio
(T1CE), operando com YOLOv11s em imagens 2D. O projeto investiga
duas variáveis metodológicas sistematicamente ignoradas na
literatura: o ratio ótimo de amostras negativas no treinamento
e a escolha arquitetural entre modelo multiclasse unificado e
especialistas binários por classe tumoral.

Os resultados foram documentados em paper acadêmico:
**"Otimização por Classe em Detecção de Tumor Cerebral com YOLO:
Ratio de Negativos e Arquitetura Multiclasse vs. Binária"**

---

## Por que isso importa

Tumores cerebrais estão entre as condições neurológicas de maior
mortalidade. Glioblastomas apresentam sobrevida mediana inferior
a 15 meses mesmo com tratamento agressivo. O diagnóstico precoce
é o principal fator de sobrevivência, mas depende de radiologistas
experientes — um recurso escasso em grande parte do mundo.

A literatura reporta mAP acima de 99% em modelos YOLO para tumor
cerebral, mas quase nenhum paper mede a taxa de falsos positivos
(FPR) em imagens saudáveis. Este projeto demonstra que um modelo
com 99% de mAP pode ter FPR de 50% — metade dos pacientes
saudáveis receberiam diagnóstico incorreto de tumor.

---

## Dataset

**BRISC 2025** (Fateh et al., arXiv:2506.14318)
6.000 imagens MRI T1CE distribuídas em 4 classes, nos três
planos anatômicos (axial, coronal, sagital).

| Classe | Imagens | Split |
|---|---|---|
| Glioma | ~1.500 | 80/10/10 estratificado |
| Meningioma | ~1.500 | por classe e plano |
| Pituitary | ~1.500 | seed=42 |
| No tumor | ~1.500 | |

Split: 4.802 treino / 599 val / 599 teste.
Masks de segmentação pixel-wise disponíveis para as 3 classes
tumorais — convertidas para bounding boxes YOLO via detecção
de contorno.

> **Nota:** o projeto iniciou com o dataset Figshare (3.064
> imagens, 233 pacientes). A abordagem 2.5D foi testada mas
> abandonada — o Figshare numera arquivos globalmente sem
> garantia de sequência volumétrica por paciente, tornando
> o empilhamento de fatias adjacentes anatomicamente inválido.
> A migração para BRISC 2025 resolveu volume, diversidade e
> a ausência de amostras negativas.

---

## Modelos treinados

### Modelo multiclasse — Run 7 (melhor)

YOLOv11s treinado nas 3 classes tumorais + no_tumor (nc=3).
Best epoch: 62/100.

| Métrica | Valor |
|---|---|
| mAP@0.50 | 0.9195 |
| mAP@0.5:0.95 | — |
| Precision | 0.9178 |
| Recall | 0.8840 |
| FPR (imagens saudáveis) | 0.50 |

### Ablation de ratio — 12 modelos binários

4 ratios (10/20/30/40%) × 3 tumores = 12 modelos (nc=1 cada).
Variável única isolada: proporção de imagens no_tumor.

**Resultados do ratio ótimo por classe:**

| Tumor | Ratio ótimo | mAP@0.50 | Recall | FPR |
|---|---|---|---|---|
| Glioma | 20% | 0.8497 | 0.8509 | 0.0% |
| Meningioma | 20% | 0.9727 | 0.9693 | 0.0% |
| Pituitary | 30% | 0.9671 | 0.9489 | 2.7% |

### Modelo de triagem

YOLOv11s treinado para detecção binária tumor/no_tumor (nc=1).

| Métrica | Valor |
|---|---|
| mAP@0.50 | 0.9425 |
| Precision | 0.9142 |
| Recall | 0.9019 |
| FPR | 0.83% |

---

## Findings principais

**1. Ratio ótimo é propriedade da classe tumoral**
Glioma e meningioma maximizam performance com 20% de negativos.
Pituitary requer 30–40% — sua anatomia de base craniana exige
maior exposição a exemplos saudáveis para suprimir FPR.
O padrão 1:1 predominante na literatura não tem justificativa
matemática para detecção com YOLO.

**2. Competição inter-classe produz efeitos assimétricos**
Glioma performa melhor no modelo multiclasse (recall 0.8214 vs.
0.7786 no binário) — sua morfologia difusa se beneficia do
contraste inter-classe. Meningioma performa melhor no especialista
binário (mAP 0.9727 vs. 0.9195 global) — sua morfologia distinta
é prejudicada pela competição com glioma.

**3. FPR é a métrica clínica que mAP esconde**
FPR de 0.50 no modelo multiclasse (clinicamente inaceitável)
cai para ~0% nos especialistas binários de glioma e meningioma,
sem nenhuma mudança arquitetural além do objetivo de classificação.

---

## Metodologia de anotação de negativos

Imagens no_tumor são anotadas com label vazio (Empty Annotations).
Quando o arquivo de label está vazio, o YOLO ignora os cálculos
de IoU-regression e DFL, disparando apenas o BCE de classificação.
É o único método arquiteturalmente válido.

Alternativas testadas e rejeitadas:
- Full-Image Bbox: destrói o DFL, FPR > 40%
- Anatomical Brain Contour: paradoxo semântico, FPR 10–15%

---

## Setup experimental

| Parâmetro | Valor |
|---|---|
| Modelo | YOLOv11s |
| Hardware | RTX 5060, 8GB VRAM |
| epochs | 100 |
| imgsz | 640 |
| batch | 16 |
| patience | 30 |
| cos_lr | True |
| amp | True |
| Pesos iniciais | COCO pretrained |

---

## Limitações conhecidas

- Dataset cobre apenas 3 tumores primários. Abscessos cerebrais
  têm aparência idêntica ao glioblastoma no T1CE e seriam
  classificados incorretamente.
- Classe no_tumor do BRISC é híbrida (saudáveis + benignos
  não-neoplásicos).
- Pituitary 20% requer rerun com patience=50 — run atual
  inconclusiva (best epoch = último epoch).
- Validação externa não realizada.

---

## Trabalho futuro

**Imediato:** rerun pituitary 20% com patience=50.

**Médio prazo:** pipeline em cascata — triagem → multiclasse
para glioma + especialistas binários para meningioma e pituitary.

**Longo prazo:** extensão para anomalias não-neoplásicas com
realce em T1CE (metástases, linfoma primário do SNC) via
datasets públicos Brain-Mets-Lung (TCIA) e UCSF-PCNSL (AWS).

---

## Stack

Python 3.12 · PyTorch · Ultralytics YOLOv11s · OpenCV ·
h5py · numpy · CUDA 12.1