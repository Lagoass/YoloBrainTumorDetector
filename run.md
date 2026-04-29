# Comandos de Execução (Cheatsheet)

**⚠️ AVISO IMPORTANTE: NÃO utilize o terminal integrado do VS Code ou o PowerShell padrão do Windows para estes comandos. O comando `conda` não é reconhecido nativamente por eles sem configurações extras.**

## 1. Configuração Inicial (Rodar apenas a primeira vez)

### Passo 0: Abrir o Terminal Correto
Vá até o Menu Iniciar do Windows, busque e abra exclusivamente o **Anaconda Prompt (Miniconda3)**.

### Passo 1: Instalação do Ambiente
Copie e cole os comandos abaixo, linha por linha, no Anaconda Prompt:

```bash
# 1. Navegar até a pasta do projeto
cd C:\Users\gulag\Documents\AiHealthCare\YoloBrainTumorDetector

# 2. Criar o ambiente isolado com Python 3.12
conda create -n yolov11 python=3.12 -y

# 3. Ativar o ambiente recém-criado
conda activate yolov11

# 4. Instalar o PyTorch com suporte a Placas de Vídeo (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Instalar o resto do ecossistema
pip install ultralytics opencv-python h5py
```

---

## 2. Como Rodar no Dia a Dia (Copy & Paste)

Para facilitar seu fluxo de trabalho, você não precisa rodar comando por comando.
Sempre que quiser executar, abra o **Anaconda Prompt (Miniconda3)** e cole o bloco desejado abaixo:

**Pipeline Completo — Treino (100 épocas) → Avaliação → Predição:**
```bash
cd C:\Users\gulag\Documents\AiHealthCare\YoloBrainTumorDetector && conda activate yolov11 && python src/pipeline.py
```

**Teste Rápido — Pipeline com 5 épocas:**
```bash
cd C:\Users\gulag\Documents\AiHealthCare\YoloBrainTumorDetector && conda activate yolov11 && python src/pipeline.py --epochs 5
```

**Apenas Avaliação + Predição — Pular treino (usa o best.pt existente):**
```bash
cd C:\Users\gulag\Documents\AiHealthCare\YoloBrainTumorDetector && conda activate yolov11 && python src/pipeline.py --skip-train
```

**Avaliação Standalone — mAP/Precision/Recall no split de teste:**
```bash
cd C:\Users\gulag\Documents\AiHealthCare\YoloBrainTumorDetector && conda activate yolov11 && python src/evaluate.py
```

**Predição Standalone — Inferência visual em 10 imagens aleatórias do teste:**
```bash
cd C:\Users\gulag\Documents\AiHealthCare\YoloBrainTumorDetector && conda activate yolov11 && python src/predict.py
```

*Nota: A sua RTX 5060 de 8GB está segura. O pipeline roda nativamente com `imgsz=640`, `batch=16` e `amp=True` para prevenir Out of Memory (OOM).*
