# M2D2: Multimodal Deception Detection Model

## Overview

M2D2 (Multimodal Deception Detection Model) is a transformer-based framework for detecting deception in audio-visual data. Using the DOLOS dataset (1,675 clips, 213 subjects), it implements trimodal fusion of visual (ViT-B/16), audio (Wav2Vec2-base), and text (BERT-base via Whisper ASR) modalities with cross-attention fusion, targeting 75-85% accuracy. The system employs lightweight Uniform Temporal Adapters for parameter-efficient fine-tuning (5-10% of 300M parameters).

**Keywords**: Deception Detection, Multimodal Learning, Transformer Networks, Cross-Modal Fusion

---

## Architecture

---

## Dataset: DOLOS

- **Clips**: 1,675 videos (2-19s, median 5s)
- **Subjects**: 213 (141 male, 72 female)
- **Labels**: Binary (truthful/deceptive)
- **Annotations**: MUMIN scheme (25 facial + 5 speech cues)
- **Splits**: 70% train (1,172) | 15% val (251) | 15% test (252)

**Preprocessing**: Frame extraction (30 FPS), audio resampling (16kHz), ASR transcription (Whisper), tokenization (BERT, max len 512)

---

## Installation

### Prerequisites

- Python 3.8+, CUDA 11.0+, 16GB RAM, 20GB disk

### Setup

---

## Usage

### Training

### Evaluation

### Inference

### Testing & Linting

---

## Project Structure

```
M2D2_Multimodal_Deception_Detection/
├── src/
│   ├── data/              # Dataset, preprocessing, ASR
│   ├── models/            # Visual/Audio/Text branches, fusion, classifier
│   ├── training/          # Trainer, losses, schedulers
│   └── utils/             # Augmentations, metrics, checkpointing
├── tests/                 # Mirror src structure
├── configs/               # YAML experiment configs
├── scripts/               # train.py, evaluate.py, inference.py
├── notebooks/             # Data exploration, analysis
├── checkpoints/           # Saved models
├── logs/                  # Training logs
├── requirements.txt       # Dependencies
└── AGENTS.md              # AI agent guidelines
```

---

## Experimental Setup

### Hardware & Software

### Hyperparameters

| Parameter    | Value | Parameter | Value |
| ------------ | ----- | --------- | ----- |
| Batch Size   |       |           |       |
| Max Epochs   |       |           |       |
| Weight Decay |       |           |       |
| Hidden Dim   |       |           |       |

### Evaluation Metrics

**Primary**: Accuracy, F1-Score, AUC-ROC  
**Secondary**: Precision, Recall, Confusion Matrix

### Protocols

---

## Expected Results

- **Accuracy**: 75-85% (5-10% over unimodal baselines)
- **Modality Contribution**: Visual (micro-expressions), Audio (prosody), Text (linguistic cues)
- **Efficiency**: 5-10% parameter fine-tuning
- **Interpretability**: Attention weights for modality importance

---

## Reproducibility

---
