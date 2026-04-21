import torch

from src.models.text_model import TextModel


class TestTextModel:
    def test_init_with_nlp_adapter(self, num_encoders):
        model = TextModel(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )
        assert model is not None
        assert len(model.TRANSFORMER) == num_encoders

    def test_init_with_conv_adapter(self, num_encoders):
        model = TextModel(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="conv",
        )
        assert model is not None
        assert len(model.TRANSFORMER) == num_encoders

    def test_init_no_adapter(self, num_encoders):
        model = TextModel(
            num_encoders=num_encoders,
            adapter=False,
            adapter_type="nlp",
        )
        assert model is not None
        assert len(model.TRANSFORMER) == num_encoders

    def test_forward_shape(self, device, batch_size, num_encoders, seq_length):
        model = TextModel(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )
        model.to(device)
        model.eval()

        input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)

        assert output.shape == (batch_size, 2)

    def test_gradient_flow_with_adapter(self, device, batch_size, num_encoders, seq_length):
        model = TextModel(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )
        model.to(device)

        input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)

        output = model(input_ids, attention_mask)
        loss = output.sum()
        loss.backward()

        adapter_grads = [p.grad for n, p in model.named_parameters() if "adapter" in n and p.requires_grad]
        has_adapter_grads = any(g is not None for g in adapter_grads)

        assert has_adapter_grads, "Adapter params should have gradients"

    def test_frozen_bert_params(self, num_encoders):
        model = TextModel(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )

        embedding_params = list(model.EMBEDDINGS.parameters())
        frozen_embeddings = [p for p in embedding_params if not p.requires_grad]

        assert len(frozen_embeddings) > 0, "BERT embedding params should be frozen"

    def test_trainable_params_with_adapter(self, num_encoders):
        model = TextModel(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )

        trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        adapter_params = [(n, p) for n, p in trainable_params if "adapter" in n]
        classifier_params = [(n, p) for n, p in trainable_params if "classifier" in n]

        assert len(adapter_params) > 0, "Adapter params should be trainable"
        assert len(classifier_params) > 0, "Classifier params should be trainable"

    def test_output_logits_valid(self, device, batch_size, num_encoders, seq_length):
        model = TextModel(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )
        model.to(device)
        model.eval()

        input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains infinite values"
