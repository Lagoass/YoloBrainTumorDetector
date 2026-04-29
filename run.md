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

## 2. Como Rodar o Treino no Dia a Dia (Copy & Paste)

Para facilitar seu fluxo de trabalho, você não precisa rodar comando por comando. 
Sempre que quiser iniciar o treinamento, abra o **Anaconda Prompt (Miniconda3)** e cole o bloco único abaixo:

**Executar Treino Completo (100 épocas):**
```bash
cd C:\Users\gulag\Documents\AiHealthCare\YoloBrainTumorDetector && conda activate yolov11 && python src/train.py
```

**Executar Teste Rápido (Apenas 1 época):**
```bash
cd C:\Users\gulag\Documents\AiHealthCare\YoloBrainTumorDetector && conda activate yolov11 && python src/train.py 1
```

*Nota: A sua RTX 5060 de 8GB está segura. O script `src/train.py` já roda nativamente com `imgsz=640`, `batch=16` e `amp=True` para prevenir Out of Memory (OOM).*
