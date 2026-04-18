# -*- coding: UTF-8 -*-
'''
@Project ：M2D2_Multimodal_Deception_Detection-main 
@File    ：inference_txt_bert_with_lora.py
@IDE     ：PyCharm 
@Author  ：Winston·H DONG
@Date    ：2026/4/18 15:09 
'''
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel

MODEL_DIR = "./src/models/bert_lie_truth_lora"
BASE_MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
base_model = BertForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME, num_labels=2
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
    return "lie" if pred == 1 else "truth"

print(predict("The only concert I have ever been to was by show."))
