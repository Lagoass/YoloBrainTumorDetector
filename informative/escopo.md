# Projeto: Localizador de Tumores Cerebrais (Edge AI 2D)

## 1. Escopo e Restrições
* **Objetivo:** Detecção e localização de tumores cerebrais via Bounding Boxes em exames de imagem médica (MRI/CT).
* **Restrição Arquitetural:** Estritamente 2D (Frame-a-Frame). Nenhuma técnica de reconstrução 3D.
* **Hardware Alvo:** NVIDIA RTX 5060 (8GB VRAM). O consumo de memória (OOM) é o principal gargalo a ser evitado.
* **Modelo Base:** YOLOv11s (Ultralytics).

## 2. Estratégia de Dados e Pré-processamento
* **Dataset Inicial (MVP):** Figshare Brain Tumor Dataset (priorizado por já ser fatiado em 2D).
* **Conversão de Formato:** Implementar script para converter as máscaras/coordenadas do Figshare para o padrão normalizado YOLO `[class_id x_center y_center width height]`.
* **Empilhamento 2.5D (Crucial):** Para contornar a falta de profundidade do 2D sem estourar a VRAM, o tensor de entrada RGB simulado deve conter:
    * Canal R = Fatia Z-1 (anterior)
    * Canal G = Fatia Z (fatia alvo com o ground truth)
    * Canal B = Fatia Z+1 (próxima)

## 3. Hiperparâmetros Obrigatórios de Treinamento (Sobrevivência em 8GB VRAM)
Qualquer script de treinamento gerado (`train.py` ou via CLI) deve IMPRETERIVELMENTE conter as seguintes configurações para o YOLO:
* `imgsz`: 640 (Não reduzir para não perder anomalias diminutas).
* `batch`: 16 (Limite matemático da VRAM nesta resolução).
* `amp`: True (Automatic Mixed Precision FP16 ativado para corte de consumo de memória e uso dos Tensor Cores Ada Lovelace).

## 4. Diretrizes de Data Augmentation Médico (.yaml)
O pipeline deve desativar aumentos sintéticos que corrompem a anatomia humana. Modificar o arquivo de hiperparâmetros com:
* `mixup`: 0.0 (Desativado).
* `flipud`: 0.0 (Desativado - preservar orientação dorso-ventral).
* `degrees`: Limite entre [-10.0, 10.0].
* `fliplr`: 0.5 (Ativado - o cérebro tem simetria bilateral).
* `mosaic`: 1.0 (Ativado), MAS garantir `close_mosaic`: 10 (desativar nas últimas 10 épocas).

## 5. Pipeline de Inferência
* **Formato de Exportação:** TensorRT (`.engine`).
* **Parâmetros de Exportação:** `format=engine half=True imgsz=640`.
* **Justificativa:** O arquivo `.pt` padrão carrega gradientes. O `.engine` funde camadas em C++, opera em FP16, libera VRAM e reduz a latência para ~5-6ms.

## 6. Próximas Tarefas para o Agente de Código
1. Analisar este contexto.
2. Escrever o script em Python de conversão do dataset Figshare para o formato de diretórios YOLO (`/images/train`, `/labels/train`, etc) aplicando o empilhamento 2.5D caso os dados permitam a sequência volumétrica.
3. Gerar o script de treinamento `train.py` consumindo os hiperparâmetros exatos do item 3 e 4.