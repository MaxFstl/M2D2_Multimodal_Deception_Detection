from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
import torchaudio
import torchvision.transforms as T


@dataclass
class VideoAugmentation:
    """
    Args:
        horizontal_flip: Random horizontal flip probability.
        color_jitter: List of [brightness, contrast, saturation, hue] adjustments.
        random_erasing_prob: Random erasing probability.
        random_erasing_scale: Random erasing scale range.
        random_rotation: Maximum rotation angle in degrees.
        apply_prob: Probability of applying augmentations to each sample.
    """

    horizontal_flip: float = 0.5
    color_jitter: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.1])
    random_erasing_prob: float = 0.0
    random_erasing_scale: tuple = (0.02, 0.2)
    random_rotation: int = 15
    apply_prob: float = 0.5

    def __post_init__(self) -> None:
        transforms = [
            T.RandomHorizontalFlip(p=self.horizontal_flip),
            T.ColorJitter(
                brightness=self.color_jitter[0],
                contrast=self.color_jitter[1],
                saturation=self.color_jitter[2],
                hue=self.color_jitter[3],
            ),
            T.RandomRotation(degrees=self.random_rotation),
        ]
        if self.random_erasing_prob > 0:
            transforms.append(
                T.RandomErasing(
                    p=self.random_erasing_prob, scale=self.random_erasing_scale
                )
            )
        self._transform = T.Compose(transforms)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.apply_prob:
            return frames

        augmented = []
        for i in range(frames.shape[0]):
            augmented.append(self._transform(frames[i]))
        return torch.stack(augmented, dim=0)


@dataclass
class AudioAugmentation:
    """
    Args:
        noise_factor: Standard deviation of additive Gaussian noise.
        time_stretch_min: Minimum time stretch ratio.
        time_stretch_max: Maximum time stretch ratio.
        pitch_shift_steps: Maximum pitch shift in semitones.
        volume_perturb_db: Maximum volume perturbation in dB.
        noise_prob: Probability of applying noise injection.
        time_stretch_prob: Probability of applying time stretch.
        pitch_shift_prob: Probability of applying pitch shift.
        volume_prob: Probability of applying volume perturbation.
        apply_prob: Overall probability of applying any augmentation.
    """

    noise_factor: float = 0.005
    time_stretch_min: float = 0.9
    time_stretch_max: float = 1.1
    pitch_shift_steps: int = 2
    volume_perturb_db: float = 2.0
    noise_prob: float = 0.5
    time_stretch_prob: float = 0.25
    pitch_shift_prob: float = 0.25
    volume_prob: float = 0.5
    apply_prob: float = 0.5

    def __call__(
        self, waveform: torch.Tensor, sample_rate: int = 16000
    ) -> torch.Tensor:
        if np.random.rand() > self.apply_prob:
            return waveform

        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        if np.random.rand() < self.noise_prob:
            waveform = self._add_noise(waveform)

        if np.random.rand() < self.time_stretch_prob:
            waveform = self._time_stretch(waveform, sample_rate)

        if np.random.rand() < self.pitch_shift_prob:
            waveform = self._pitch_shift(waveform, sample_rate)

        if np.random.rand() < self.volume_prob:
            waveform = self._volume_perturb(waveform)

        return waveform.unsqueeze(0)

    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform) * self.noise_factor
        return waveform + noise

    def _time_stretch(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        stretch_factor = np.random.uniform(self.time_stretch_min, self.time_stretch_max)
        return torchaudio.functional.time_stretch(
            waveform, orig_freq=sample_rate, factor=stretch_factor
        )

    def _pitch_shift(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        n_steps = np.random.randint(-self.pitch_shift_steps, self.pitch_shift_steps + 1)
        return torchaudio.functional.pitch_shift(
            waveform, sample_rate=sample_rate, n_steps=n_steps
        )

    def _volume_perturb(self, waveform: torch.Tensor) -> torch.Tensor:
        db_change = np.random.uniform(-self.volume_perturb_db, self.volume_perturb_db)
        amplitude_ratio = 10 ** (db_change / 20)
        return waveform * amplitude_ratio


def create_video_augmentation(config: Dict[str, Any]) -> VideoAugmentation:
    return VideoAugmentation(
        horizontal_flip=config.get("horizontal_flip", 0.5),
        color_jitter=config.get("color_jitter", [0.2, 0.2, 0.2, 0.1]),
        random_erasing_prob=config.get("random_erasing_prob", 0.0),
        random_erasing_scale=tuple(config.get("random_erasing_scale", [0.02, 0.2])),
        random_rotation=config.get("random_rotation", 15),
        apply_prob=config.get("apply_prob", 0.5),
    )


def create_audio_augmentation(config: Dict[str, Any]) -> AudioAugmentation:
    return AudioAugmentation(
        noise_factor=config.get("noise_factor", 0.005),
        time_stretch_min=config.get("time_stretch_min", 0.9),
        time_stretch_max=config.get("time_stretch_max", 1.1),
        pitch_shift_steps=config.get("pitch_shift_steps", 2),
        volume_perturb_db=config.get("volume_perturb_db", 2.0),
        noise_prob=config.get("noise_prob", 0.5),
        time_stretch_prob=config.get("time_stretch_prob", 0.25),
        pitch_shift_prob=config.get("pitch_shift_prob", 0.25),
        volume_prob=config.get("volume_prob", 0.5),
        apply_prob=config.get("apply_prob", 0.5),
    )


def create_augmentations(config: Dict[str, Any]) -> tuple:
    aug_config = config.get("augmentation", {})
    video_aug = create_video_augmentation(aug_config.get("video", {}))
    audio_aug = create_audio_augmentation(aug_config.get("audio", {}))
    return video_aug, audio_aug
