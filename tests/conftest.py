import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def num_encoders():
    return 2


@pytest.fixture
def num_frames():
    return 16


@pytest.fixture
def audio_length():
    return 16000


@pytest.fixture
def seq_length():
    return 32
