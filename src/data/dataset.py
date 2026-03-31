# FROM: https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning/blob/main/scripts/dataloader/audio_visual_dataset.py

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from scipy.io import wavfile
from scipy import signal
from transformers import BertTokenizer


def _get_tokenizer() -> BertTokenizer:
    return BertTokenizer.from_pretrained("bert-base-uncased")


class DOLOSDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        audio_dir: str,
        img_dir: str,
        texts: Optional[List[str]] = None,
        num_frames: int = 16,
        frame_size: int = 224,
        num_audio_tokens: int = 64,
        use_mock_text: bool = True,
        text_max_length: int = 512,
    ):
        super().__init__()
        self.annos = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.img_dir = img_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_audio_tokens = num_audio_tokens
        self.use_mock_text = use_mock_text
        self.text_max_length = text_max_length

        self.texts = texts if texts is not None else [None] * len(self.annos)

        self.video_transform = T.Compose(
            [
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Resize((frame_size, frame_size)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.annos)

    def _get_mock_transcript(self, clip_name: str) -> str:
        templates = [
            "I am absolutely certain that {} is completely true.",
            "Let me explain what happened with {}.",
            "The reason I {} is because of circumstances.",
            "I want to be completely honest about {}.",
            "Without a doubt, {} happened exactly this way.",
        ]
        objects = [
            "the package",
            "the meeting",
            "the transaction",
            "the agreement",
            "the conversation",
        ]
        template = random.choice(templates)
        obj = random.choice(objects)
        return template.format(obj)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_name = self.annos.iloc[idx, 1]

        audio = self._load_audio(clip_name)
        video = self._load_video(clip_name)
        text = self._load_text(idx, clip_name)
        label = self._parse_label(idx)

        return {
            "audio": audio,
            "video": video,
            "text": text,
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _load_audio(self, clip_name: str) -> torch.Tensor:
        audio_path = os.path.join(self.audio_dir, clip_name + ".wav")
        sample_rate, waveform = wavfile.read(audio_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        clip_duration = waveform.size(1) / sample_rate
        new_sample_rate = int(321.893491124260 * self.num_audio_tokens / clip_duration)
        num_samples = int(clip_duration * new_sample_rate)
        waveform = waveform.squeeze(0).numpy()
        waveform = signal.resample(waveform, num_samples)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        return waveform.float()

    def _load_video(self, clip_name: str) -> torch.Tensor:
        file_path = os.path.join(self.img_dir, clip_name + "/")
        frame_names = sorted([i for i in os.listdir(file_path) if i.endswith(".jpg")])

        target_frames = np.linspace(0, len(frame_names) - 1, num=self.num_frames)
        target_frames = np.round(target_frames).astype(int)

        face_frames = []
        for i in target_frames:
            img = (
                np.asarray(Image.open(os.path.join(file_path, frame_names[i]))) / 255.0
            )
            face_frames.append(self.video_transform(img))
        return torch.stack(face_frames, 0).float()

    def _load_text(self, idx: int, clip_name: str) -> str:
        if self.use_mock_text or self.texts[idx] is None:
            return self._get_mock_transcript(clip_name)
        return self.texts[idx]

    def _parse_label(self, idx: int) -> int:
        str_label = self.annos.iloc[idx, -1]
        if str_label == "truth":
            return 0
        elif str_label == "deception":
            return 1
        raise ValueError(f"Undefined label: {str_label}")


def multimodal_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    audio_list, video_list, text_list, label_list = [], [], [], []

    for item in batch:
        audio_list.append(item["audio"])
        video_list.append(item["video"])
        text_list.append(item["text"])
        label_list.append(item["label"])

    max_audio_len = max(a.size(1) for a in audio_list)
    audio_padded = torch.zeros(len(audio_list), 1, max_audio_len)
    for i, a in enumerate(audio_list):
        audio_padded[i, :, : a.size(1)] = a

    video_tensor = torch.stack(video_list)

    encoded = _get_tokenizer()(
        text_list, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )

    labels = torch.stack(label_list)

    return {
        "audio": audio_padded,
        "video": video_tensor,
        "text_ids": encoded["input_ids"],
        "text_mask": encoded["attention_mask"],
        "label": labels,
    }
