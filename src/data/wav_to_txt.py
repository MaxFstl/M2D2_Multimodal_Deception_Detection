# -*- coding: UTF-8 -*-
'''
@Project ：M2D2_Multimodal_Deception_Detection-main 
@File    ：wav_to_txt.py
@IDE     ：PyCharm 
@Author  ：Winston·H DONG
@Date    ：2026/4/18 13:56 
'''
import os
import whisper

model = whisper.load_model("large-v3")  # medium / large-v3

audio_dir = "../../data/DOLOS/audio_files"
out_dir = "../../data/DOLOS/txt_files"
os.makedirs(out_dir, exist_ok=True)

for name in os.listdir(audio_dir):
    if not name.lower().endswith(".wav"):
        continue
    path = os.path.join(audio_dir, name)
    print("Transcribing:", path)
    result = model.transcribe(path, language="en")  # language="zh"
    txt_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
