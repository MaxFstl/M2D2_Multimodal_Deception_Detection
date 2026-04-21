import os
import pytest

from src.utils.load_config import load_config


class TestLoadConfig:
    def test_load_default_config(self):
        config = load_config()
        assert config is not None
        assert isinstance(config, dict)

    def test_config_has_required_keys(self):
        config = load_config()
        required_keys = ["model", "training", "data", "paths"]
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_config_training_params(self):
        config = load_config()
        assert "batch_size" in config["training"]
        assert "learning_rate" in config["training"]
        assert "num_epochs" in config["training"]

    def test_config_data_params(self):
        config = load_config()
        assert "train_split" in config["data"]
        assert "val_split" in config["data"]
        assert "test_split" in config["data"]

    def test_config_paths_exist(self):
        config = load_config()
        assert "annotations" in config["paths"]
        assert "audio_dir" in config["paths"]
        assert "image_dir" in config["paths"]

    def test_missing_config_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_config_splits_sum_to_one(self):
        config = load_config()
        total = (
            config["data"]["train_split"]
            + config["data"]["val_split"]
            + config["data"]["test_split"]
        )
        assert abs(total - 1.0) < 0.01
