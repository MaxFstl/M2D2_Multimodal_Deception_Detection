from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import DOLOSDataset, multimodal_collate_fn
from src.data.validation import validate_clips
from src.utils.logging import get_logger

logger = get_logger(__name__)


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
    original_count = len(annos)
    
    valid_clips, missing_clips = validate_clips(annotations_file, audio_dir, img_dir)
    
    if missing_clips:
        annos = annos[annos.iloc[:, 1].isin(valid_clips)].reset_index(drop=True)
        logger.info(
            f"Filtered {len(missing_clips)} missing clips for dataloader creation, "
            f"keeping {len(annos)}/{original_count} clips"
        )
    
    clip_names = annos.iloc[:, 1].tolist()
    labels = annos.iloc[:, -1].apply(lambda x: 0 if x == "truth" else 1).tolist()

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

    train_clips = set(clip_names[i] for i in train_idx)
    val_clips = set(clip_names[i] for i in val_idx)
    test_clips = set(clip_names[i] for i in test_idx)
    
    train_texts = {k: v for k, v in texts.items() if k in train_clips} if texts else None
    val_texts = {k: v for k, v in texts.items() if k in val_clips} if texts else None
    test_texts = {k: v for k, v in texts.items() if k in test_clips} if texts else None

    train_annos = annos[annos.iloc[:, 1].isin(train_clips)].reset_index(drop=True)
    val_annos = annos[annos.iloc[:, 1].isin(val_clips)].reset_index(drop=True)
    test_annos = annos[annos.iloc[:, 1].isin(test_clips)].reset_index(drop=True)

    train_dataset = DOLOSDataset(
        annotations=train_annos,
        audio_dir=audio_dir,
        img_dir=img_dir,
        texts=train_texts,
        num_frames=config.get("num_frames", 16),
        frame_size=config.get("frame_size", 224),
        num_audio_tokens=config.get("num_audio_tokens", 64),
        use_mock_text=use_mock_text,
        skip_validation=True,
    )

    val_dataset = DOLOSDataset(
        annotations=val_annos,
        audio_dir=audio_dir,
        img_dir=img_dir,
        texts=val_texts,
        num_frames=config.get("num_frames", 16),
        frame_size=config.get("frame_size", 224),
        num_audio_tokens=config.get("num_audio_tokens", 64),
        use_mock_text=use_mock_text,
        skip_validation=True,
    )

    test_dataset = DOLOSDataset(
        annotations=test_annos,
        audio_dir=audio_dir,
        img_dir=img_dir,
        texts=test_texts,
        num_frames=config.get("num_frames", 16),
        frame_size=config.get("frame_size", 224),
        num_audio_tokens=config.get("num_audio_tokens", 64),
        use_mock_text=use_mock_text,
        skip_validation=True,
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
