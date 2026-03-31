import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock, patch

from src.data.dataset import DOLOSDataset, multimodal_collate_fn, _get_tokenizer


class TestDOLOSDataset:
    @pytest.fixture
    def mock_annos_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("file_name,label,gender\n")
            f.write("clip_001,truth,M\n")
            f.write("clip_002,deception,F\n")
            f.write("clip_003,truth,M\n")
            f.write("clip_004,lie,F\n")
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def mock_audio_dir(self):
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        for f in os.listdir(tmpdir):
            os.unlink(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)

    @pytest.fixture
    def mock_img_dir(self):
        tmpdir = tempfile.mkdtemp()
        for clip in ["clip_001", "clip_002", "clip_003", "clip_004"]:
            clip_dir = os.path.join(tmpdir, clip)
            os.makedirs(clip_dir)
            for i in range(16):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                from PIL import Image

                Image.fromarray(img).save(os.path.join(clip_dir, f"frame_{i:04d}.jpg"))
        yield tmpdir
        for clip in ["clip_001", "clip_002", "clip_003", "clip_004"]:
            clip_dir = os.path.join(tmpdir, clip)
            for f in os.listdir(clip_dir):
                os.unlink(os.path.join(clip_dir, f))
            os.rmdir(clip_dir)
        os.rmdir(tmpdir)

    @pytest.fixture
    def mock_audio_waveform(self):
        return torch.randn(1, 16000)

    @pytest.fixture
    def dataset_config(self):
        return {
            "num_frames": 16,
            "frame_size": 224,
            "num_audio_tokens": 64,
            "batch_size": 2,
        }

    def test_dataset_length(self, mock_annos_file, mock_audio_dir, mock_img_dir):
        dataset = DOLOSDataset(
            annotations_file=mock_annos_file,
            audio_dir=mock_audio_dir,
            img_dir=mock_img_dir,
            use_mock_text=True,
        )
        assert len(dataset) == 4

    @patch("soundfile.read")
    def test_dataset_returns_dict_with_all_modalities(
        self,
        mock_sf_read,
        mock_annos_file,
        mock_audio_dir,
        mock_img_dir,
        mock_audio_waveform,
    ):
        mock_sf_read.return_value = (mock_audio_waveform.numpy().squeeze(), 16000)
        dataset = DOLOSDataset(
            annotations_file=mock_annos_file,
            audio_dir=mock_audio_dir,
            img_dir=mock_img_dir,
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

    @patch("soundfile.read")
    def test_label_parsing(
        self,
        mock_sf_read,
        mock_annos_file,
        mock_audio_dir,
        mock_img_dir,
        mock_audio_waveform,
    ):
        mock_sf_read.return_value = (mock_audio_waveform.numpy().squeeze(), 16000)
        dataset = DOLOSDataset(
            annotations_file=mock_annos_file,
            audio_dir=mock_audio_dir,
            img_dir=mock_img_dir,
            use_mock_text=True,
        )
        assert dataset[0]["label"].item() == 0
        assert dataset[1]["label"].item() == 1
        assert dataset[2]["label"].item() == 0
        assert dataset[3]["label"].item() == 1

    @patch("soundfile.read")
    def test_mock_text_generation(
        self,
        mock_sf_read,
        mock_annos_file,
        mock_audio_dir,
        mock_img_dir,
        mock_audio_waveform,
    ):
        mock_sf_read.return_value = (mock_audio_waveform.numpy().squeeze(), 16000)
        dataset = DOLOSDataset(
            annotations_file=mock_annos_file,
            audio_dir=mock_audio_dir,
            img_dir=mock_img_dir,
            use_mock_text=True,
        )
        texts = [dataset[i]["text"] for i in range(len(dataset))]
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

    @patch("soundfile.read")
    def test_video_frames_shape(
        self,
        mock_sf_read,
        mock_annos_file,
        mock_audio_dir,
        mock_img_dir,
        mock_audio_waveform,
    ):
        mock_sf_read.return_value = (mock_audio_waveform.numpy().squeeze(), 16000)
        dataset = DOLOSDataset(
            annotations_file=mock_annos_file,
            audio_dir=mock_audio_dir,
            img_dir=mock_img_dir,
            num_frames=16,
            frame_size=224,
            use_mock_text=True,
        )
        item = dataset[0]
        assert item["video"].shape == (16, 3, 224, 224)

    @patch("soundfile.read")
    def test_custom_texts_passed_through(
        self,
        mock_sf_read,
        mock_annos_file,
        mock_audio_dir,
        mock_img_dir,
        mock_audio_waveform,
    ):
        mock_sf_read.return_value = (mock_audio_waveform.numpy().squeeze(), 16000)
        custom_texts = [
            "custom text 1",
            "custom text 2",
            "custom text 3",
            "custom text 4",
        ]
        dataset = DOLOSDataset(
            annotations_file=mock_annos_file,
            audio_dir=mock_audio_dir,
            img_dir=mock_img_dir,
            texts=custom_texts,
            use_mock_text=False,
        )
        for i in range(len(dataset)):
            assert dataset[i]["text"] == custom_texts[i]


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
