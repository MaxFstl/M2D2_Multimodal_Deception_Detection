import os
import whisper
from src.utils.load_config import load_config
import torch


def transcribeAudio(audio_dir, out_dir):
    if os.path.exists(out_dir):
        existing_files = [f for f in os.listdir(out_dir)]
        print(f"Found {len(existing_files)} existing files in {out_dir}")

        if len(existing_files) > 0:
            print("Already extracted text. Skipping this step.")
            return
        else:
            print("Directory exists but empty. Preprocessing the text...")

    model = whisper.load_model("medium")  # medium / large-v3
    config = load_config()
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
