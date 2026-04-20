import pytest
import torch
import torchaudio
import torchvision
from transformers import BertModel

from src.models.adapter import (
    Efficient_Conv_Pass,
    w2v2_adapter_nlp,
    w2v2_adapter_conv,
    vit_adapter_nlp,
    vit_adapter_conv,
    bert_adapter_nlp,
    bert_adapter_conv,
)


class TestEfficientConvPass:
    def test_shape_preserved(self, device, batch_size, seq_length):
        conv_pass = Efficient_Conv_Pass()
        conv_pass.to(device)
        conv_pass.eval()

        x = torch.randn(batch_size, seq_length, 768, device=device)

        with torch.no_grad():
            output = conv_pass(x)

        assert output.shape == x.shape

    def test_gradient_flow(self, device, batch_size, seq_length):
        conv_pass = Efficient_Conv_Pass()
        conv_pass.to(device)

        x = torch.randn(batch_size, seq_length, 768, device=device, requires_grad=True)
        output = conv_pass(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in conv_pass.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestW2V2Adapters:
    @pytest.fixture
    def w2v2_encoder_layer(self):
        model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        return model.encoder.transformer.layers[0]

    def test_nlp_adapter_forward(self, w2v2_encoder_layer, device, batch_size):
        adapter = w2v2_adapter_nlp(w2v2_encoder_layer)
        adapter.to(device)
        adapter.eval()

        x = torch.randn(batch_size, 64, 768, device=device)

        with torch.no_grad():
            output = adapter(x)

        assert output.shape == x.shape

    def test_conv_adapter_forward(self, w2v2_encoder_layer, device, batch_size):
        adapter = w2v2_adapter_conv(w2v2_encoder_layer)
        adapter.to(device)
        adapter.eval()

        x = torch.randn(batch_size, 64, 768, device=device)

        with torch.no_grad():
            output = adapter(x)

        assert output.shape == x.shape


class TestViTAdapters:
    @pytest.fixture
    def vit_encoder_layer(self):
        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        return vit_b_16.encoder.layers[0]

    def test_nlp_adapter_forward(self, vit_encoder_layer, device, batch_size):
        adapter = vit_adapter_nlp(vit_encoder_layer)
        adapter.to(device)
        adapter.eval()

        x = torch.randn(batch_size, 16, 768, device=device)

        with torch.no_grad():
            output = adapter(x)

        assert output.shape == x.shape

    def test_conv_adapter_forward(self, vit_encoder_layer, device, batch_size):
        adapter = vit_adapter_conv(vit_encoder_layer)
        adapter.to(device)
        adapter.eval()

        x = torch.randn(batch_size, 16, 768, device=device)

        with torch.no_grad():
            output = adapter(x)

        assert output.shape == x.shape


class TestBertAdapters:
    @pytest.fixture
    def bert_encoder_layer(self):
        bert = BertModel.from_pretrained("bert-base-uncased")
        return bert.encoder.layer[0]

    def test_nlp_adapter_forward(self, bert_encoder_layer, device, batch_size, seq_length):
        adapter = bert_adapter_nlp(bert_encoder_layer)
        adapter.to(device)
        adapter.eval()

        x = torch.randn(batch_size, seq_length, 768, device=device)

        with torch.no_grad():
            output = adapter(x)

        assert output.shape == x.shape

    def test_conv_adapter_forward(self, bert_encoder_layer, device, batch_size, seq_length):
        adapter = bert_adapter_conv(bert_encoder_layer)
        adapter.to(device)
        adapter.eval()

        x = torch.randn(batch_size, seq_length, 768, device=device)

        with torch.no_grad():
            output = adapter(x)

        assert output.shape == x.shape

    def test_adapter_param_count(self, bert_encoder_layer):
        adapter = bert_adapter_nlp(bert_encoder_layer)

        adapter_params = sum(p.numel() for p in adapter.adapter.parameters())

        assert adapter_params < 200000, f"Adapter has too many params: {adapter_params}"


class TestAdapterTrainability:
    def test_nlp_adapter_params_trainable(self):
        bert = BertModel.from_pretrained("bert-base-uncased")
        adapter = bert_adapter_nlp(bert.encoder.layer[0])

        trainable_params = [n for n, p in adapter.named_parameters() if p.requires_grad]
        adapter_trainable = [n for n in trainable_params if "adapter" in n]

        assert len(adapter_trainable) > 0, "No trainable adapter params found"

    def test_frozen_base_params(self):
        bert = BertModel.from_pretrained("bert-base-uncased")
        
        frozen_before = [p for p in bert.parameters() if not p.requires_grad]
        assert len(frozen_before) == 0, "BERT params start as trainable"
        
        for p in bert.parameters():
            p.requires_grad = False
            
        adapter = bert_adapter_nlp(bert.encoder.layer[0])
        
        attention_params = [p for n, p in adapter.named_parameters() if "attention" in n]
        frozen_attention = [p for p in attention_params if not p.requires_grad]

        assert len(frozen_attention) > 0, "Attention params should be frozen"
