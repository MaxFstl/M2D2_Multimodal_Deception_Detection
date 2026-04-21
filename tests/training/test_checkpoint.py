import pytest
import torch
import torch.nn as nn

from src.training.train_test import save_checkpoint, load_checkpoint


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class TestCheckpointSaveLoad:
    @pytest.fixture
    def model(self):
        return MockModel()

    @pytest.fixture
    def optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.001)

    def test_save_checkpoint_creates_file(self, model, optimizer, tmp_path):
        checkpoint_path = save_checkpoint(
            checkpoint_dir=str(tmp_path),
            model_name="test_model",
            model=model,
            optimizer=optimizer,
            epoch=5,
            best_acc=0.85,
            config={"test": True},
        )
        assert tmp_path.exists()
        saved_file = tmp_path / "test_model_epoch_5.pt"
        assert saved_file.exists()

    def test_save_best_checkpoint_creates_both(self, model, optimizer, tmp_path):
        checkpoint_path = save_checkpoint(
            checkpoint_dir=str(tmp_path),
            model_name="test_model",
            model=model,
            optimizer=optimizer,
            epoch=5,
            best_acc=0.85,
            config={"test": True},
            is_best=True,
        )
        best_file = tmp_path / "test_model_best.pt"
        epoch_file = tmp_path / "test_model_epoch_5.pt"
        assert best_file.exists()
        assert epoch_file.exists()

    def test_load_checkpoint_restores_model(self, model, optimizer, tmp_path):
        save_checkpoint(
            checkpoint_dir=str(tmp_path),
            model_name="test_model",
            model=model,
            optimizer=optimizer,
            epoch=5,
            best_acc=0.85,
            config={"test": True},
        )
        new_model = MockModel()
        checkpoint_path = str(tmp_path / "test_model_epoch_5.pt")
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        epoch, best_acc, config = load_checkpoint(
            checkpoint_path, new_model, new_optimizer
        )
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.allclose(param1, param2)

    def test_load_checkpoint_restores_optimizer(self, model, optimizer, tmp_path):
        optimizer.step()
        save_checkpoint(
            checkpoint_dir=str(tmp_path),
            model_name="test_model",
            model=model,
            optimizer=optimizer,
            epoch=5,
            best_acc=0.85,
            config={"test": True},
        )
        new_model = MockModel()
        checkpoint_path = str(tmp_path / "test_model_epoch_5.pt")
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        load_checkpoint(checkpoint_path, new_model, new_optimizer)
        orig_groups = optimizer.state_dict()["param_groups"]
        new_groups = new_optimizer.state_dict()["param_groups"]
        assert orig_groups[0]["lr"] == new_groups[0]["lr"]

    def test_load_checkpoint_returns_metadata(self, model, optimizer, tmp_path):
        save_checkpoint(
            checkpoint_dir=str(tmp_path),
            model_name="test_model",
            model=model,
            optimizer=optimizer,
            epoch=10,
            best_acc=0.92,
            config={"num_encoders": 4, "adapter": True},
        )
        new_model = MockModel()
        checkpoint_path = str(tmp_path / "test_model_epoch_10.pt")
        epoch, best_acc, config = load_checkpoint(checkpoint_path, new_model)
        assert epoch == 10
        assert best_acc == 0.92
        assert config["num_encoders"] == 4
        assert config["adapter"] is True
