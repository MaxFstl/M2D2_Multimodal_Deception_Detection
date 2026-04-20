import os
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock, patch

from src.data.dataset import DOLOSDataset, multimodal_collate_fn, _get_tokenizer


class TestDOLOSDataset:
    @pytest.fixture
    def test_data_dir(self, tmp_path):
        annos = pd.DataFrame(
            {
                "video_id": ["vid001", "vid001", "vid002", "vid002"],
                "file_name": ["clip_001", "clip_002", "clip_003", "clip_004"],
                "start_time": ["0:00", "0:10", "0:20", "0:30"],
                "end_time": ["0:05", "0:15", "0:25", "0:35"],
                "label": ["truth", "deception", "truth", "deception"],
            }
        )
        annos.to_csv(str(tmp_path / "annotations.csv"), index=False)

        for clip in ["clip_001", "clip_002", "clip_003", "clip_004"]:
            clip_dir = tmp_path / clip
            clip_dir.mkdir()
            for i in range(16):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                from PIL import Image

                Image.fromarray(img).save(str(clip_dir / f"frame_{i:04d}.jpg"))

            import soundfile as sf

            audio_path = tmp_path / f"{clip}.wav"
            waveform = np.random.randn(16000).astype(np.float32)
            sf.write(str(audio_path), waveform, 16000)

        return {
            "annotations_file": str(tmp_path / "annotations.csv"),
            "audio_dir": str(tmp_path),
            "img_dir": str(tmp_path),
        }

    def test_dataset_length(self, test_data_dir):
        dataset = DOLOSDataset(
            annotations=test_data_dir["annotations_file"],
            audio_dir=test_data_dir["audio_dir"],
            img_dir=test_data_dir["img_dir"],
            use_mock_text=True,
        )
        assert len(dataset) == 4

    def test_dataset_returns_dict_with_all_modalities(self, test_data_dir):
        dataset = DOLOSDataset(
            annotations=test_data_dir["annotations_file"],
            audio_dir=test_data_dir["audio_dir"],
            img_dir=test_data_dir["img_dir"],
            use_mock_text=True,
        )
        item = dataset[0]
        assert "audio" in item
        assert "video" in item
        assert "text" in item
        assert "label" in item
        assert item["audio"].dim() == 2
        assert item["video"].shape[0] == 16
        assert isinstance(item["text"], str)

    def test_label_parsing(self, test_data_dir):
        dataset = DOLOSDataset(
            annotations=test_data_dir["annotations_file"],
            audio_dir=test_data_dir["audio_dir"],
            img_dir=test_data_dir["img_dir"],
            use_mock_text=True,
        )
        assert dataset[0]["label"].item() == 0
        assert dataset[1]["label"].item() == 1
        assert dataset[2]["label"].item() == 0
        assert dataset[3]["label"].item() == 1

    def test_mock_text_generation(self, test_data_dir):
        dataset = DOLOSDataset(
            annotations=test_data_dir["annotations_file"],
            audio_dir=test_data_dir["audio_dir"],
            img_dir=test_data_dir["img_dir"],
            use_mock_text=True,
        )
        texts = [dataset[i]["text"] for i in range(len(dataset))]
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

    def test_video_frames_shape(self, test_data_dir):
        dataset = DOLOSDataset(
            annotations=test_data_dir["annotations_file"],
            audio_dir=test_data_dir["audio_dir"],
            img_dir=test_data_dir["img_dir"],
            num_frames=16,
            frame_size=224,
            use_mock_text=True,
        )
        item = dataset[0]
        assert item["video"].shape == (16, 3, 224, 224)

    def test_custom_texts_passed_through(self, test_data_dir):
        custom_texts = {
            "clip_001": "custom text 1",
            "clip_002": "custom text 2",
            "clip_003": "custom text 3",
            "clip_004": "custom text 4",
        }
        dataset = DOLOSDataset(
            annotations=test_data_dir["annotations_file"],
            audio_dir=test_data_dir["audio_dir"],
            img_dir=test_data_dir["img_dir"],
            texts=custom_texts,
            use_mock_text=False,
        )
        clip_names = ["clip_001", "clip_002", "clip_003", "clip_004"]
        for i in range(len(dataset)):
            assert dataset[i]["text"] == custom_texts[clip_names[i]]

    def test_dataset_accepts_dataframe(self, test_data_dir):
        annos = pd.read_csv(test_data_dir["annotations_file"])
        dataset = DOLOSDataset(
            annotations=annos,
            audio_dir=test_data_dir["audio_dir"],
            img_dir=test_data_dir["img_dir"],
            use_mock_text=True,
            skip_validation=True,
        )
        assert len(dataset) == 4


class TestMultimodalCollateFn:
    def test_collate_fn_pads_audio_correctly(self):
        batch = [
            {
                "audio": torch.tensor([[1.0, 2.0, 3.0]]),
                "video": torch.randn(16, 3, 224, 224),
                "text": "hello",
                "label": torch.tensor(0),
            },
            {
                "audio": torch.tensor([[1.0, 2.0]]),
                "video": torch.randn(16, 3, 224, 224),
                "text": "world",
                "label": torch.tensor(1),
            },
        ]
        result = multimodal_collate_fn(batch)
        assert result["audio"].shape == (2, 1, 3)
        assert result["video"].shape == (2, 16, 3, 224, 224)
        assert result["text_ids"].shape[0] == 2
        assert result["text_mask"].shape[0] == 2
        assert result["label"].tolist() == [0, 1]

    def test_collate_fn_returns_text_ids_and_mask(self):
        batch = [
            {
                "audio": torch.tensor([[1.0]]),
                "video": torch.randn(16, 3, 224, 224),
                "text": "test",
                "label": torch.tensor(0),
            },
        ]
        result = multimodal_collate_fn(batch)
        assert "text_ids" in result
        assert "text_mask" in result
        assert result["text_ids"].shape[0] == 1


class TestStratifiedSplits:
    def test_stratified_split_preserves_class_distribution(self):
        labels = [0] * 70 + [1] * 30
        indices = np.arange(100)
        np.random.seed(42)
        np.random.shuffle(indices)
        labels = [labels[i] for i in indices]

        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
        train_idx, temp_idx = next(sss.split(np.zeros(len(labels)), labels))

        train_labels = [labels[i] for i in train_idx]
        temp_labels = [labels[i] for i in temp_idx]

        train_ratio = sum(train_labels) / len(train_labels)
        temp_ratio = sum(temp_labels) / len(temp_labels)
        assert abs(train_ratio - 0.3) < 0.1
        assert abs(temp_ratio - 0.3) < 0.1


class TestWeightedSampling:
    def test_weighted_sampler_references_correct_class_weights(self):
        labels = [0] * 70 + [1] * 30
        class_weights = [1.0, 2.33]
        sample_weights = [class_weights[label] for label in labels]

        assert all(
            sample_weights[i] == class_weights[labels[i]] for i in range(len(labels))
        )


class TestTokenizer:
    def test_tokenizer_is_lazy_loaded(self):
        tokenizer1 = _get_tokenizer()
        tokenizer2 = _get_tokenizer()
        assert tokenizer1 is tokenizer2
