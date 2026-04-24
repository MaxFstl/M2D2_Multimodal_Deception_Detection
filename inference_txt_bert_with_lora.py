# -*- coding: UTF-8 -*-
'''
@Project ：M2D2_Multimodal_Deception_Detection-main 
@File    ：inference_txt_bert_with_lora.py
@IDE     ：PyCharm 
@Author  ：Winston·H DONG
@Date    ：2026/4/18 15:09 
'''
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

MODEL_DIR = "./bert_lie_truth_lora"
BASE_MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

base_model = BertForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    num_labels=2
)

model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.to(DEVICE)
model.eval()

# label convention:
# 0 = truth
# 1 = lie
id2label = {
    0: "truth",
    1: "lie"
}

label2id = {
    "truth": 0,
    "lie": 1
}


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()

    return id2label[pred]


def evaluate_test_split(test_texts, test_labels, batch_size=16):
    """
    test_texts: list[str]
    test_labels: list[int] or list[str]
                 int labels should be 0=truth, 1=lie
                 string labels should be "truth" or "lie"
    """

    # Convert string labels to ids if needed
    if isinstance(test_labels[0], str):
        test_labels = [label2id[x.lower()] for x in test_labels]

    all_preds = []
    all_labels = []

    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=1)
    precision = precision_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)

    print("===== Test Split Evaluation =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    print("\n===== Classification Report =====")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["truth", "lie"],
        digits=4
    ))

    print("===== Confusion Matrix =====")
    print(confusion_matrix(all_labels, all_preds))
    print("Rows = true labels, Columns = predicted labels")
    print("[[truth->truth, truth->lie],")
    print(" [lie->truth,   lie->lie]]")

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "predictions": all_preds,
        "labels": all_labels
    }


# -----------------------------
# Single prediction examples
# -----------------------------

print(predict("The only concert I have ever been to was by show."))
# content from file AN_WILTY_EP21_lie18.txt
