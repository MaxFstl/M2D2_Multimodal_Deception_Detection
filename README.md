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
- [uv](https://github.com/astral-sh/uv) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Setup

```bash
uv sync
```

Download the videos:

```
uv run -m scripts.download_data
```

---

## Usage

### Training

### Evaluation

### Inference

### Testing & Linting

```bash
pytest                    # Run all tests
pytest -v                 # Verbose
pytest -k "pattern"       # Pattern matching
ruff check .              # Linter
ruff format .             # Formatter
mypy src/                 # Type checking
```

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

## Development Progress

### Phase 1: Project Setup

- [x] Create directory structure (src/data, src/models, src/training, src/utils, tests, configs, scripts, notebooks, checkpoints, logs)
- [x] Create **init**.py files for all Python packages
- [x] Create pyproject.toml with dependencies (torch, transformers, pandas, numpy, etc.)
- [x] Create base config YAML file (configs/default.yaml) with model/training/data parameters

### Phase 2: Data Pipeline

- [x] Automatically download the data (scripts/download_data.py) -> uv run -m scripts.download_data.py
- [x] Implement dataset loader (src/data/dataset.py) - DOLOS video/audio/text loading
- [x] Implement video preprocessing (src/data/utils/face) - frame extraction, normalization
- [x] Implement audio preprocessing (src/data/utils/video_to_audio.py)
- [ ] Implement ASR transcription module (src/data/asr.py) - Whisper integration
- [ ] Implement text preprocessing (src/data/text_utils.py) - BERT tokenization, max len 512
- [x] Implement data transforms/augmentations (src/utils/augmentations.py)
- [x] Create DataLoader with stratified splits (src/data/loaders.py)
- [x] Write unit tests for data loading pipeline

### Phase 3: Models

- [ ] Implement visual encoder branch (src/models/visual.py) - ViT-B/16 + UT-Adapter
- [ ] Implement audio encoder branch (src/models/audio.py) - Wav2Vec2 + UT-Adapter
- [ ] Implement text encoder branch (src/models/text.py) - BERT + adapter
- [ ] Implement cross-attention fusion module (src/models/fusion.py) - 2-4 heads, dim 768
- [ ] Implement classifier head (src/models/classifier.py) - MLP to [B, 512] -> binary
- [ ] Implement main M2D2 model (src/models/m2d2.py) - combine all components
- [ ] Write unit tests for model components

### Phase 4: Training

- [ ] Implement training loop (src/training/trainer.py) - epoch iteration, logging
- [ ] Implement loss functions (src/training/losses.py) - weighted cross-entropy for imbalance
- [ ] Implement optimizer & scheduler (src/training/optim.py) - AdamW, lr=1e-4
- [ ] Implement checkpointing (src/utils/checkpoint.py) - save/load model state
- [ ] Implement early stopping (src/training/early_stopping.py)
- [ ] Implement metrics tracking (src/utils/metrics.py) - accuracy, F1, AUC-ROC
- [ ] Create train.py script (scripts/train.py) - CLI interface

### Phase 5: Evaluation

- [ ] Implement evaluation pipeline (src/training/evaluate.py)
- [ ] Create evaluate.py script (scripts/evaluate.py) - checkpoint loading, metrics
- [ ] Implement confusion matrix and classification report generation

### Phase 6: Inference

- [ ] Implement inference pipeline (src/utils/inference.py)
- [ ] Create inference.py script (scripts/inference.py) - single clip prediction

### Phase 7: Utils

- [ ] Implement seed setting for reproducibility (src/utils/reproducibility.py)
- [ ] Implement logging utilities (src/utils/logging_utils.py)

### Phase 8: Testing

- [ ] Write comprehensive test suite for all modules
- [ ] Set up pytest configuration and conftest.py fixtures
- [ ] Run all tests and achieve >80% coverage

### Phase 9: Documentation

- [ ] Update README.md with complete usage instructions
- [ ] Add docstrings to all modules, classes, and functions

### Phase 10: Final

- [ ] Run linting and type checking (ruff, mypy)
- [ ] Create experiment configs for different hyperparameter settings
- [ ] End-to-end integration test with sample data

---

## Reproducibility

---

## Remarks

LLMs were used to generate this README and generate a raw plan of the project
