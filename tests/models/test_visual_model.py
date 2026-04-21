import torch

from src.models.visual_model import conv2d_block, cnn_face, ViT_model


class TestConv2dBlock:
    def test_output_shape(self, device, batch_size):
        block = conv2d_block(in_channels=3, out_channels=64, k=3, s=1)
        block.to(device)
        block.eval()

        x = torch.randn(batch_size, 3, 64, 64, device=device)

        with torch.no_grad():
            output = block(x)

        assert output.shape == (batch_size, 64, 64, 64)


class TestCnnFace:
    def test_output_shape(self, device, batch_size):
        cnn = cnn_face()
        cnn.to(device)
        cnn.eval()

        x = torch.randn(batch_size, 3, 160, 160, device=device)

        with torch.no_grad():
            output = cnn(x)

        assert output.shape == (batch_size, 256, 1, 1)

    def test_squeezed_output(self, device, batch_size):
        cnn = cnn_face()
        cnn.to(device)
        cnn.eval()

        x = torch.randn(batch_size, 3, 160, 160, device=device)

        with torch.no_grad():
            output = cnn(x)
            squeezed = output.squeeze(-1).squeeze(-1)

        assert squeezed.shape == (batch_size, 256)


class TestViTModel:
    def test_init_with_nlp_adapter(self, num_encoders):
        model = ViT_model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )
        assert model is not None

    def test_init_with_conv_adapter(self, num_encoders):
        model = ViT_model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="conv",
        )
        assert model is not None

    def test_init_no_adapter(self, num_encoders):
        model = ViT_model(
            num_encoders=num_encoders,
            adapter=False,
            adapter_type="nlp",
        )
        assert model is not None

    def test_forward_shape(self, device, batch_size, num_encoders):
        model = ViT_model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )
        model.to(device)
        model.eval()

        visual_input = torch.randn(batch_size, 64, 3, 160, 160, device=device)

        with torch.no_grad():
            output = model(visual_input)

        assert output.shape == (batch_size, 2)

    def test_gradient_flow_with_adapter(self, device, batch_size, num_encoders):
        model = ViT_model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )
        model.to(device)

        visual_input = torch.randn(batch_size, 64, 3, 160, 160, device=device)
        output = model(visual_input)
        loss = output.sum()
        loss.backward()

        adapter_grads = [p.grad for n, p in model.named_parameters() if "adapter" in n and p.requires_grad]
        has_adapter_grads = any(g is not None for g in adapter_grads)

        assert has_adapter_grads, "Adapter params should have gradients"

    def test_frozen_vit_params(self, num_encoders):
        model = ViT_model(
            num_encoders=num_encoders,
            adapter=True,
            adapter_type="nlp",
        )

        vit_encoder_params = list(model.ViT_Encoder.parameters())
        frozen_vit = [p for p in vit_encoder_params if not p.requires_grad]

        assert len(frozen_vit) > 0, "Some ViT encoder params should be frozen"
