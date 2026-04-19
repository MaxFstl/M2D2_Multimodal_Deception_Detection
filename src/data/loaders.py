from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import DOLOSDataset, multimodal_collate_fn


def create_dataloaders(
    annotations_file: str,
    audio_dir: str,
    img_dir: str,
    config: Dict,
    texts: Optional[Dict[str, str]] = None,
    use_mock_text: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    annos = pd.read_csv(annotations_file)
    clip_names = annos.iloc[:, 1].tolist()
    labels = annos.iloc[:, -1].apply(lambda x: 0 if x == "truth" else 1).tolist()

    existing_audio = {
        f.replace(".wav", "") for f in os.listdir(audio_dir) if f.endswith(".wav")
    }
    existing_video = {
        d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))
    }
    valid_clips = existing_audio & existing_video

    train_split = config.get("train_split", 0.7)
    val_split = config.get("val_split", 0.15)
    test_split = config.get("test_split", 0.15)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_split, random_state=42)
    train_idx, temp_idx = next(sss.split(np.zeros(len(labels)), labels))

    temp_labels = [labels[i] for i in temp_idx]
    val_size = val_split / (val_split + test_split)
    sss_val = StratifiedShuffleSplit(n_splits=1, train_size=val_size, random_state=42)
    val_idx_local, test_idx_local = next(
        sss_val.split(np.zeros(len(temp_labels)), temp_labels)
    )
    val_idx = [temp_idx[i] for i in val_idx_local]
    test_idx = [temp_idx[i] for i in test_idx_local]

    train_labels = [labels[i] for i in train_idx]
    class_weights = config.get("class_weights", [1.0, 1.5])
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    def get_texts_for_indices(indices: List[int]) -> Optional[List[str]]:
        if texts is None:
            return None
        return [texts.get(clip_names[i]) for i in indices]

    train_dataset = DOLOSDataset(
        annotations_file=annotations_file,
        audio_dir=audio_dir,
        img_dir=img_dir,
        texts=get_texts_for_indices(train_idx),
        num_frames=config.get("num_frames", 16),
        frame_size=config.get("frame_size", 224),
        num_audio_tokens=config.get("num_audio_tokens", 64),
        use_mock_text=use_mock_text,
        valid_clips=valid_clips,
    )

    val_dataset = DOLOSDataset(
        annotations_file=annotations_file,
        audio_dir=audio_dir,
        img_dir=img_dir,
        texts=get_texts_for_indices(val_idx),
        num_frames=config.get("num_frames", 16),
        frame_size=config.get("frame_size", 224),
        num_audio_tokens=config.get("num_audio_tokens", 64),
        use_mock_text=use_mock_text,
        valid_clips=valid_clips,
    )

    test_dataset = DOLOSDataset(
        annotations_file=annotations_file,
        audio_dir=audio_dir,
        img_dir=img_dir,
        texts=get_texts_for_indices(test_idx),
        num_frames=config.get("num_frames", 16),
        frame_size=config.get("frame_size", 224),
        num_audio_tokens=config.get("num_audio_tokens", 64),
        use_mock_text=use_mock_text,
        valid_clips=valid_clips,
    )

    batch_size = config.get("batch_size", 16)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multimodal_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multimodal_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multimodal_collate_fn,
    )

    return train_loader, val_loader, test_loader
