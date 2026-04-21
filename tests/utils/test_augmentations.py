import numpy as np
import pytest
import torch

from src.utils.augmentations import (
    VideoAugmentation,
    AudioAugmentation,
    create_video_augmentation,
    create_audio_augmentation,
)


class TestVideoAugmentation:
    @pytest.fixture
    def sample_frames(self):
        return torch.randn(16, 3, 224, 224)

    def test_output_shape_preserved(self, sample_frames):
        aug = VideoAugmentation(apply_prob=1.0)
        output = aug(sample_frames)
        assert output.shape == sample_frames.shape

    def test_no_augmentation_when_apply_prob_zero(self, sample_frames):
        aug = VideoAugmentation(apply_prob=0.0)
        output = aug(sample_frames)
        assert torch.allclose(output, sample_frames)

    def test_augmentation_applied_stochastically(self, sample_frames):
        aug = VideoAugmentation(apply_prob=0.5)
        outputs = [aug(sample_frames) for _ in range(20)]
        changed = [not torch.allclose(o, sample_frames) for o in outputs]
        assert 0.1 < sum(changed) / len(changed) < 0.9

    def test_horizontal_flip_only(self, sample_frames):
        aug = VideoAugmentation(
            horizontal_flip=1.0,
            color_jitter=[0.0, 0.0, 0.0, 0.0],
            random_rotation=0,
            apply_prob=1.0,
        )
        output = aug(sample_frames)
        assert output.shape == sample_frames.shape

    def test_create_video_augmentation_from_config(self):
        config = {
            "horizontal_flip": 0.3,
            "color_jitter": [0.1, 0.1, 0.1, 0.05],
            "random_rotation": 10,
            "apply_prob": 0.8,
        }
        aug = create_video_augmentation(config)
        assert aug.horizontal_flip == 0.3
        assert aug.random_rotation == 10
        assert aug.apply_prob == 0.8


class TestAudioAugmentation:
    @pytest.fixture
    def sample_waveform(self):
        return torch.randn(1, 16000)

    def test_output_is_tensor(self, sample_waveform):
        aug = AudioAugmentation(apply_prob=1.0)
        output = aug(sample_waveform)
        assert isinstance(output, torch.Tensor)

    def test_no_augmentation_when_apply_prob_zero(self, sample_waveform):
        aug = AudioAugmentation(apply_prob=0.0)
        output = aug(sample_waveform)
        assert torch.allclose(output, sample_waveform)

    def test_noise_addition_changes_waveform(self, sample_waveform):
        aug = AudioAugmentation(
            noise_factor=0.1,
            noise_prob=1.0,
            time_stretch_prob=0.0,
            pitch_shift_prob=0.0,
            volume_prob=0.0,
            apply_prob=1.0,
        )
        output = aug(sample_waveform.clone())
        assert not torch.allclose(output, sample_waveform)

    def test_volume_perturbation_changes_amplitude(self, sample_waveform):
        aug = AudioAugmentation(
            volume_perturb_db=10.0, volume_prob=1.0, apply_prob=1.0
        )
        output = aug(sample_waveform.clone())
        orig_energy = torch.sum(sample_waveform**2).item()
        new_energy = torch.sum(output**2).item()
        assert abs(new_energy - orig_energy) > 0.01 * orig_energy

    def test_create_audio_augmentation_from_config(self):
        config = {
            "noise_factor": 0.01,
            "pitch_shift_steps": 3,
            "volume_perturb_db": 5.0,
            "apply_prob": 0.7,
        }
        aug = create_audio_augmentation(config)
        assert aug.noise_factor == 0.01
        assert aug.pitch_shift_steps == 3
        assert aug.volume_perturb_db == 5.0
        assert aug.apply_prob == 0.7


class TestAugmentationDeterminism:
    def test_zero_prob_is_deterministic(self):
        frames = torch.randn(16, 3, 224, 224)
        waveform = torch.randn(1, 16000)
        
        video_aug = VideoAugmentation(apply_prob=0.0)
        audio_aug = AudioAugmentation(apply_prob=0.0)
        
        for _ in range(5):
            assert torch.allclose(video_aug(frames), frames)
            assert torch.allclose(audio_aug(waveform), waveform)
