# Localizador de Tumores Cerebrais — Edge AI 2D

## O que é este projeto

Um sistema de detecção e localização automática de tumores cerebrais em exames de MRI, operando via bounding boxes em imagens 2D. O modelo identifica e classifica três tipos de tumor — meningioma, glioma e tumor hipofisário — frame a frame, sem reconstrução volumétrica 3D.

---

## Por que isso importa

Tumores cerebrais estão entre as condições neurológicas de maior mortalidade. O diagnóstico precoce é o principal fator de sobrevivência, mas depende da disponibilidade de radiologistas experientes — um recurso escasso em grande parte do mundo.

Um modelo de detecção automática que opere em hardware acessível pode atuar como triagem de primeira linha, acelerando o fluxo diagnóstico e reduzindo o tempo entre exame e tratamento.

---

## O dataset

3064 imagens de MRI T1 com contraste de gadolínio, provenientes de 233 pacientes, coletadas entre 2005 e 2010 em dois hospitais na China. Resolução de 512×512px.

| Tipo de Tumor | Fatias | Representação |
|---|---|---|
| Meningioma | 708 | 23.1% |
| Glioma | 1426 | 46.5% |
| Tumor Hipofisário | 930 | 30.4% |

O desbalanceamento entre classes — glioma com quase o dobro de amostras que meningioma — é o principal desafio de treinamento do projeto.

---

## Desafios técnicos

**Restrição de hardware** — o modelo opera em uma GPU de 8GB de VRAM. Toda a arquitetura de treinamento foi otimizada para este limite sem sacrificar resolução ou qualidade das detecções.

**Contexto volumétrico em 2D** — MRI é naturalmente tridimensional, mas processar volumes 3D exigiria muito mais memória. A solução adotada é o empilhamento 2.5D: cada imagem de entrada é um RGB sintético composto pela fatia alvo e suas duas fatias adjacentes, dando ao modelo contexto de profundidade sem custo de 3D.

**Class imbalance** — gliomas são quase 2x mais frequentes que meningiomas no dataset. Sem correção, o modelo aprende a favorecer glioma em situações de incerteza, gerando falsos positivos. O projeto aborda isso via Focal Loss, que penaliza automaticamente exemplos fáceis e força o aprendizado nas classes sub-representadas.

**Augmentation médico** — técnicas padrão de data augmentation podem corromper anatomia humana. Rotações são limitadas a ±10°, flip vertical é desativado para preservar orientação dorso-ventral, e mixup é desativado pois combinar cérebros de pacientes diferentes não representa nenhuma realidade clínica.

---

## Modelo e resultados

**Modelo:** YOLOv11s (Ultralytics), fine-tuned sobre pesos pré-treinados COCO.

**Run 1 — baseline:**
| Métrica | Valor |
|---|---|
| mAP@0.50 | 0.9217 |
| mAP@0.5:0.95 | 0.5468 |
| Precision | 0.9244 |
| Recall | 0.8453 |

---

## Stack

Python 3.12 · PyTorch · Ultralytics YOLOv11s · OpenCV · h5py · CUDA 12.1 · TensorRT

---

## Deployment

O modelo é exportado para TensorRT (`.engine`) em FP16, reduzindo latência para ~5-6ms por frame e liberando VRAM para inferência em produção.