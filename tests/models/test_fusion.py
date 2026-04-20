import torch

from src.models.fusion_model import CrossFusionModule, Fusion


class TestCrossFusionModule:
    def test_output_shape(self, device, batch_size, seq_length):
        cross_fusion = CrossFusionModule(dim=256)
        cross_fusion.to(device)
        cross_fusion.eval()

        mod1_feat = torch.randn(batch_size, seq_length, 768, device=device)
        mod2_feat = torch.randn(batch_size, seq_length, 768, device=device)

        with torch.no_grad():
            output = cross_fusion(mod1_feat, mod2_feat)

        assert output.shape == (batch_size, seq_length, 64)

    def test_gradient_flow(self, device, batch_size, seq_length):
        cross_fusion = CrossFusionModule(dim=256)
        cross_fusion.to(device)

        mod1_feat = torch.randn(batch_size, seq_length, 768, device=device)
        mod2_feat = torch.randn(batch_size, seq_length, 768, device=device)

        output = cross_fusion(mod1_feat, mod2_feat)
        loss = output.sum()
        loss.backward()

        for name, param in cross_fusion.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestFusionModel:
    def test_concat_init(self, num_encoders):
        model = Fusion(
            fusion_type="concat",
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
            multi=False,
        )
        assert model is not None
        assert model.fusion_type == "concat"

    def test_cross2_init(self, num_encoders):
        model = Fusion(
            fusion_type="cross2",
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
            multi=False,
        )
        assert model is not None
        assert model.fusion_type == "cross2"

    def test_concat_forward_shape(
        self, device, batch_size, num_encoders, num_frames, audio_length, seq_length
    ):
        model = Fusion(
            fusion_type="concat",
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
            multi=False,
            num_frames=num_frames,
        )
        model.to(device)
        model.eval()

        audio_input = torch.randn(batch_size, audio_length, device=device)
        visual_input = torch.randn(batch_size, num_frames, 3, 160, 160, device=device)
        input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)

        with torch.no_grad():
            output, _, _, _ = model(audio_input, visual_input, input_ids, attention_mask)

        assert output.shape == (batch_size, 2)

    def test_cross2_forward_shape(
        self, device, batch_size, num_encoders, num_frames, audio_length, seq_length
    ):
        model = Fusion(
            fusion_type="cross2",
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
            multi=False,
            num_frames=num_frames,
        )
        model.to(device)
        model.eval()

        audio_input = torch.randn(batch_size, audio_length, device=device)
        visual_input = torch.randn(batch_size, num_frames, 3, 160, 160, device=device)
        input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)

        with torch.no_grad():
            output, _, _, _ = model(audio_input, visual_input, input_ids, attention_mask)

        assert output.shape == (batch_size, 2)

    def test_multi_task_output(
        self, device, batch_size, num_encoders, num_frames, audio_length, seq_length
    ):
        model = Fusion(
            fusion_type="cross2",
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
            multi=True,
            num_frames=num_frames,
        )
        model.to(device)
        model.eval()

        audio_input = torch.randn(batch_size, audio_length, device=device)
        visual_input = torch.randn(batch_size, num_frames, 3, 160, 160, device=device)
        input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)

        with torch.no_grad():
            main_out, audio_out, visual_out, text_out = model(
                audio_input, visual_input, input_ids, attention_mask
            )

        assert main_out.shape == (batch_size, 2)
        assert audio_out.shape == (batch_size, 2)
        assert visual_out.shape == (batch_size, 2)
        assert text_out.shape == (batch_size, 2)

    def test_gradient_flow_concat(
        self, device, batch_size, num_encoders, num_frames, audio_length, seq_length
    ):
        model = Fusion(
            fusion_type="concat",
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
            multi=False,
            num_frames=num_frames,
        )
        model.to(device)

        audio_input = torch.randn(batch_size, audio_length, device=device)
        visual_input = torch.randn(batch_size, num_frames, 3, 160, 160, device=device)
        input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)

        output, _, _, _ = model(audio_input, visual_input, input_ids, attention_mask)
        loss = output.sum()
        loss.backward()

        adapter_grads = [p.grad for n, p in model.named_parameters() if "adapter" in n and p.requires_grad]
        has_adapter_grads = any(g is not None for g in adapter_grads)

        assert has_adapter_grads, "Adapter params should have gradients"

    def test_frozen_base_params(self, num_encoders):
        model = Fusion(
            fusion_type="concat",
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
            multi=False,
        )

        feature_extractor_params = list(model.FEATURE_EXTRACTOR.parameters())
        frozen_fe = [p for p in feature_extractor_params if not p.requires_grad]

        assert len(frozen_fe) > 0, "Feature extractor params should be frozen"

    def test_output_valid(
        self, device, batch_size, num_encoders, num_frames, audio_length, seq_length
    ):
        model = Fusion(
            fusion_type="concat",
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
            multi=False,
            num_frames=num_frames,
        )
        model.to(device)
        model.eval()

        audio_input = torch.randn(batch_size, audio_length, device=device)
        visual_input = torch.randn(batch_size, num_frames, 3, 160, 160, device=device)
        input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)

        with torch.no_grad():
            output, _, _, _ = model(audio_input, visual_input, input_ids, attention_mask)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains infinite values"
