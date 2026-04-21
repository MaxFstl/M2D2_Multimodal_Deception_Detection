import os
from typing import List, Set, Tuple

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_clips(
    annotations_file: str,
    audio_dir: str,
    img_dir: str,
) -> Tuple[Set[str], List[str]]:
    existing_audio = {
        f.replace(".wav", "")
        for f in os.listdir(audio_dir)
        if f.endswith(".wav")
    }
    
    existing_video = {
        d
        for d in os.listdir(img_dir)
        if os.path.isdir(os.path.join(img_dir, d))
    }
    
    existing_clips = existing_audio & existing_video
    
    annos = pd.read_csv(annotations_file)
    annotated_clips = set(annos.iloc[:, 1].tolist())
    
    missing_clips = sorted(annotated_clips - existing_clips)
    
    if missing_clips:
        logger.warning(
            f"Found {len(missing_clips)} missing clips out of {len(annotated_clips)} annotated"
        )
        for clip in missing_clips:
            logger.warning(f"Missing clip: {clip}")
    
    valid_clips = annotated_clips & existing_clips
    
    return valid_clips, missing_clips
