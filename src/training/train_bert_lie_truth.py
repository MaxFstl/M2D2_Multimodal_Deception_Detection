# -*- coding: UTF-8 -*-
"""
@Project ：M2D2_Multimodal_Deception_Detection-main
@File    ：train_bert_lie_truth.py
@IDE     ：PyCharm
@Author  ：Winston·H DONG
@Date    ：2026/4/18 15:05
"""

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from peft import LoraConfig, get_peft_model, TaskType


# =========================
# Config
# =========================

DATA_DIR = r"../../data/DOLOS/txt_files/txt_LV3"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "../../src/models/bert_lie_truth_lora"

MAX_LEN = 256
RANDOM_SEED = 42

# Label convention:
# 0 = truth
# 1 = lie
ID2LABEL = {
    0: "truth",
    1: "lie",
}

LABEL2ID = {
    "truth": 0,
    "lie": 1,
}


# =========================
# Dataset Class
# =========================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
        )

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            k: torch.tensor(v[idx])
            for k, v in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )

        return item

    def __len__(self):
        return len(self.labels)


# =========================
# Load Data
# =========================

def load_data(data_dir):
    texts = []
    labels = []  # 0 = truth, 1 = lie

    for fname in os.listdir(data_dir):
        if not fname.endswith(".txt"):
            continue

        lower_name = fname.lower()

        if "lie" in lower_name:
            label = 1
        elif "truth" in lower_name:
            label = 0
        else:
            continue

        path = os.path.join(data_dir, fname)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if not text:
            continue

        texts.append(text)
        labels.append(label)

    print(f"Loaded {len(texts)} samples.")
    print("Full label distribution:", Counter(labels))
    print("Label meaning: 0 = truth, 1 = lie")

    return texts, labels


# =========================
# Split Data
# =========================

def split_data(texts, labels):
    """
    Split data into:
    70% train
    15% validation
    15% test
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print("\n===== Dataset Split =====")
    print(f"Train samples     : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples      : {len(X_test)}")

    print("\n===== Label Distribution =====")
    print("Train:", Counter(y_train))
    print("Val  :", Counter(y_val))
    print("Test :", Counter(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================
# Compute Metrics
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =========================
# Build Model
# =========================

def build_model():
    base_model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,

        # More conservative LoRA setting for small dataset
        r=4,
        lora_alpha=8,
        lora_dropout=0.2,

        target_modules=["query", "value"],
    )

    model = get_peft_model(base_model, lora_config)

    print("\n===== Trainable Parameters =====")
    model.print_trainable_parameters()

    return model


# =========================
# Detailed Evaluation
# =========================

def detailed_evaluation(trainer, dataset, split_name):
    """
    Print accuracy, precision, recall, F1,
    classification report, and confusion matrix.
    """

    print(f"\n===== {split_name} Evaluation =====")

    results = trainer.evaluate(
        eval_dataset=dataset,
        metric_key_prefix=split_name.lower(),
    )


    prefix = split_name.lower()

    print(f"{split_name} Accuracy : {results[f'{prefix}_accuracy']:.4f}")
    print(f"{split_name} Precision: {results[f'{prefix}_precision']:.4f}")
    print(f"{split_name} Recall   : {results[f'{prefix}_recall']:.4f}")
    print(f"{split_name} F1       : {results[f'{prefix}_f1']:.4f}")

    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print(f"\n===== {split_name} Classification Report =====")
    print(
        classification_report(
            labels,
            preds,
            target_names=["truth", "lie"],
            digits=4,
            zero_division=0,
        )
    )

    print(f"===== {split_name} Confusion Matrix =====")
    print(confusion_matrix(labels, preds))
    print("Rows = true labels, Columns = predicted labels")
    print("[[truth->truth, truth->lie],")
    print(" [lie->truth,   lie->lie]]")

    return results


# =========================
# Main
# =========================

def main():
    # 1. Load data
    texts, labels = load_data(DATA_DIR)

    # 2. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        texts,
        labels,
    )

    # 3. Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 4. Build datasets
    train_dataset = TextDataset(
        X_train,
        y_train,
        tokenizer,
        max_len=MAX_LEN,
    )

    val_dataset = TextDataset(
        X_val,
        y_val,
        tokenizer,
        max_len=MAX_LEN,
    )

    test_dataset = TextDataset(
        X_test,
        y_test,
        tokenizer,
        max_len=MAX_LEN,
    )

    # 5. Build BERT + LoRA model
    model = build_model()

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Max epoch number.
        # Early stopping will stop training earlier if validation F1 does not improve.
        num_train_epochs=50,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,

        # Lower learning rate to reduce overfitting
        learning_rate=5e-5,
        weight_decay=0.05,

        logging_steps=10,

        # For newer transformers versions
        eval_strategy="epoch",
        save_strategy="epoch",

        # Load the checkpoint with best validation F1
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        save_total_limit=2,

        fp16=torch.cuda.is_available(),

        report_to="none",

        # Disable notebook progress callback issue
        disable_tqdm=True,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=15,
            )
        ],
    )

    # 8. Train
    print("\n===== Start Training =====")
    trainer.train()

    print("\n===== Best Validation Model Loaded =====")

    # 9. Evaluate validation split
    detailed_evaluation(
        trainer=trainer,
        dataset=val_dataset,
        split_name="Validation",
    )

    # 10. Evaluate test split
    detailed_evaluation(
        trainer=trainer,
        dataset=test_dataset,
        split_name="Test",
    )

    # 11. Save LoRA model and tokenizer
    print("\n===== Saving Model =====")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"LoRA model and tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()