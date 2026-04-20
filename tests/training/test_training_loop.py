import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.train_test import (
    train_one_epoch,
    val_one_epoch,
    evaluation,
)


class MockArgs:
    def __init__(
        self, model_to_train="fusion", device="cpu", multi=False
    ):
        self.model_to_train = model_to_train
        self.device = torch.device(device)
        self.multi = multi


class MockFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 2)

    def forward(self, audio, video, text_ids, attention_mask):
        batch_size = audio.shape[0]
        dummy_features = torch.randn(batch_size, 768)
        return (
            self.linear(dummy_features),
            self.linear(dummy_features),
            self.linear(dummy_features),
            self.linear(dummy_features),
        )


def create_mock_batch(batch_size=2):
    return {
        "audio": torch.randn(batch_size, 1, 16000),
        "video": torch.randn(batch_size, 16, 3, 224, 224),
        "text_ids": torch.randint(0, 30000, (batch_size, 32)),
        "text_mask": torch.ones(batch_size, 32, dtype=torch.long),
        "label": torch.randint(0, 2, (batch_size,)),
    }


class MockDataLoader:
    def __init__(self, num_batches=3, batch_size=2):
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            yield create_mock_batch(self.batch_size)

    def __len__(self):
        return self.num_batches


class TestTrainOneEpoch:
    @pytest.fixture
    def model(self):
        model = MockFusionModel()
        return model

    @pytest.fixture
    def optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.001)

    @pytest.fixture
    def loss_fn(self):
        return nn.CrossEntropyLoss()

    def test_returns_loss_predictions_labels(self, model, optimizer, loss_fn):
        args = MockArgs()
        train_loader = MockDataLoader(num_batches=2)
        loss, preds, labels, _, _ = train_one_epoch(
            args,
            train_loader,
            model,
            optimizer,
            loss_fn,
            None,
            None,
            None,
        )
        assert isinstance(loss, float)
        assert isinstance(preds, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert preds.shape == labels.shape

    def test_loss_is_finite(self, model, optimizer, loss_fn):
        args = MockArgs()
        train_loader = MockDataLoader(num_batches=2)
        loss, _, _, _, _ = train_one_epoch(
            args,
            train_loader,
            model,
            optimizer,
            loss_fn,
            None,
            None,
            None,
        )
        assert not np.isnan(loss)
        assert not np.isinf(loss)

    def test_model_in_train_mode(self, model, optimizer, loss_fn):
        args = MockArgs()
        train_loader = MockDataLoader(num_batches=1)
        train_one_epoch(
            args, train_loader, model, optimizer, loss_fn, None, None, None
        )
        assert model.training

    def test_gradients_flow_to_params(self, model, optimizer, loss_fn):
        args = MockArgs()
        train_loader = MockDataLoader(num_batches=1)
        initial_params = {n: p.clone() for n, p in model.named_parameters()}
        train_one_epoch(
            args, train_loader, model, optimizer, loss_fn, None, None, None
        )
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break
        assert params_changed


class TestValOneEpoch:
    @pytest.fixture
    def model(self):
        return MockFusionModel()

    @pytest.fixture
    def loss_fn(self):
        return nn.CrossEntropyLoss()

    def test_no_gradients_computed(self, model, loss_fn):
        args = MockArgs()
        val_loader = MockDataLoader(num_batches=2)
        param_grads_before = {
            n: p.grad.clone() if p.grad is not None else None
            for n, p in model.named_parameters()
        }
        with torch.no_grad():
            val_one_epoch(
                args, val_loader, model, loss_fn, None, None, None
            )
        for param in model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    def test_model_in_eval_mode(self, model, loss_fn):
        args = MockArgs()
        val_loader = MockDataLoader(num_batches=1)
        val_one_epoch(args, val_loader, model, loss_fn, None, None, None)
        assert not model.training

    def test_returns_correct_outputs(self, model, loss_fn):
        args = MockArgs()
        val_loader = MockDataLoader(num_batches=2, batch_size=4)
        loss, preds, labels, _, _ = val_one_epoch(
            args, val_loader, model, loss_fn, None, None, None
        )
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert preds.shape[0] == labels.shape[0] == 8


class TestEvaluation:
    def test_evaluation_returns_acc_f1_auc(self):
        labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        preds = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        acc, f1, auc_score = evaluation(labels, preds)
        assert isinstance(acc, float)
        assert isinstance(f1, float)
        assert isinstance(auc_score, float)
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= f1 <= 1.0
        assert 0.0 <= auc_score <= 1.0

    def test_evaluation_with_perfect_predictions(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 1, 1])
        acc, f1, auc_score = evaluation(labels, preds)
        assert acc == 1.0
        assert f1 == 1.0

    def test_evaluation_with_random_predictions(self):
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100)
        preds = np.random.randint(0, 2, 100)
        acc, f1, auc_score = evaluation(labels, preds)
        assert 0.3 <= acc <= 0.7
        assert 0.3 <= f1 <= 0.7
