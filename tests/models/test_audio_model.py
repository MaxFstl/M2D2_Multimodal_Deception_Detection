import torch

from src.models.audio_model import W2V2_Model


class TestW2V2Model:
    def test_init_with_nlp_adapter(self, num_encoders):
        model = W2V2_Model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )
        assert model is not None
        assert len(model.TRANSFORMER) == num_encoders

    def test_init_with_conv_adapter(self, num_encoders):
        model = W2V2_Model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="conv",
        )
        assert model is not None
        assert len(model.TRANSFORMER) == num_encoders

    def test_init_no_adapter(self, num_encoders):
        model = W2V2_Model(
            num_encoders=num_encoders,
            adapter=False,
            adapter_type="nlp",
        )
        assert model is not None
        assert len(model.TRANSFORMER) == num_encoders

    def test_forward_shape(self, device, batch_size, num_encoders, audio_length):
        model = W2V2_Model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )
        model.to(device)
        model.eval()

        audio_input = torch.randn(batch_size, audio_length, device=device)

        with torch.no_grad():
            output = model(audio_input)

        assert output.shape == (batch_size, 2)

    def test_frozen_base_params(self, num_encoders):
        model = W2V2_Model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )

        feature_extractor_params = list(model.FEATURE_EXTRACTOR.parameters())
        frozen_fe = [p for p in feature_extractor_params if not p.requires_grad]

        assert len(frozen_fe) > 0, "Feature extractor params should be frozen"

    def test_adapter_params_trainable(self, num_encoders):
        model = W2V2_Model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )

        trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        adapter_params = [(n, p) for n, p in trainable_params if "adapter" in n]
        classifier_params = [(n, p) for n, p in trainable_params if "classifier" in n]

        assert len(adapter_params) > 0, "Adapter params should be trainable"
        assert len(classifier_params) > 0, "Classifier params should be trainable"

    def test_no_adapter_all_encoder_trainable(self, num_encoders):
        model = W2V2_Model(
            num_encoders=num_encoders,
            adapter=False,
            adapter_type="nlp",
        )

        transformer_params = list(model.TRANSFORMER.parameters())
        trainable_transformer = [p for p in transformer_params if p.requires_grad]

        assert len(trainable_transformer) > 0, "Transformer params should be trainable without adapter"

    def test_output_logits_valid(self, device, batch_size, num_encoders, audio_length):
        model = W2V2_Model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )
        model.to(device)
        model.eval()

        audio_input = torch.randn(batch_size, audio_length, device=device)

        with torch.no_grad():
            output = model(audio_input)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains infinite values"
