# FROM: https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning/blob/main/scripts/models/fusion_model.py
import torch
import torch.nn as nn
import torchaudio
import torchvision
from transformers import BertModel
from src.models.adapter import (
    w2v2_adapter_nlp,
    w2v2_adapter_conv,
    vit_adapter_nlp,
    vit_adapter_conv,
    bert_adapter_nlp,
    bert_adapter_conv,
)
from src.models.visual_model import cnn_face
from torch.nn import functional as F


class CrossFusionModule(nn.Module):
    """
    crossmodal fusion module: to get crossmodal attention and to return the fused feature.
    The cross attention is calculated by the output embedding from each encoder layers of two modalities.
    """

    def __init__(self, dim=256):
        super(CrossFusionModule, self).__init__()

        self.project_mod1 = nn.Linear(768, dim)
        self.project_mod2 = nn.Linear(768, dim)
        self.corr_weights = torch.nn.Parameter(
            torch.empty(dim, dim, requires_grad=True).type(torch.cuda.FloatTensor)
        )
        nn.init.xavier_normal_(self.corr_weights)
        self.project_bottleneck = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.LayerNorm((64,), eps=1e-05, elementwise_affine=True),
            nn.ReLU(),
        )

    def forward(self, mod1_feat, mod2_feat):
        mod1_feat = self.project_mod1(mod1_feat)
        mod2_feat = self.project_mod2(mod2_feat)

        mod2_feat = mod2_feat.transpose(1, 2)

        m1 = torch.matmul(mod1_feat, self.corr_weights)
        cc_mat = torch.bmm(m1, mod2_feat)

        mod1_att = F.softmax(cc_mat, dim=1)
        mod2_att = F.softmax(cc_mat.transpose(1, 2), dim=1)
        atten_mod1features = torch.bmm(mod1_feat.transpose(1, 2), mod1_att)
        atten_mod2features = torch.bmm(mod2_feat, mod2_att)
        atten_mod1features = atten_mod1features + mod1_feat.transpose(1, 2)
        atten_mod2features = atten_mod2features + mod2_feat

        fused_features = self.project_bottleneck(
            torch.cat((atten_mod1features, atten_mod2features), dim=1).transpose(
                1, 2
            )
        )

        return fused_features


class Fusion(nn.Module):
    """
    Trimodal fusion model: audio + visual + text
    Supports fusion types: "concat" or "cross2"
    """

    def __init__(
        self,
        fusion_type,
        num_encoders,
        adapter,
        adapter_type,
        multi=False,
        num_frames: int = 16,
    ):
        super(Fusion, self).__init__()
        self.fusion_type = fusion_type
        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type
        self.multi = multi
        self.num_frames = num_frames

        # ========== AUDIO MODEL ==========
        audio_model = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        for p in audio_model.parameters():
            p.requires_grad = False

        self.FEATURE_EXTRACTOR = audio_model.feature_extractor

        self.FEATURE_PROJECTOR = nn.Sequential(
            audio_model.encoder.feature_projection,
            audio_model.encoder.transformer.pos_conv_embed,
            audio_model.encoder.transformer.layer_norm,
            audio_model.encoder.transformer.dropout,
        )

        audio_layer_list = []
        for i in range(self.num_encoders):
            if self.adapter:
                if self.adapter_type == "nlp":
                    audio_layer_list.append(
                        w2v2_adapter_nlp(
                            transformer_encoder=audio_model.encoder.transformer.layers[i]
                        )
                    )
                else:
                    audio_layer_list.append(
                        w2v2_adapter_conv(
                            transformer_encoder=audio_model.encoder.transformer.layers[i]
                        )
                    )
            else:
                for p in audio_model.encoder.transformer.layers[i].parameters():
                    p.requires_grad = True
                audio_layer_list.append(audio_model.encoder.transformer.layers[i])

        self.AUDIO_TRANSFORMER = nn.Sequential(*audio_layer_list)

        # ========== VISUAL MODEL ==========
        self.projection = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU(),
        )

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = False

        vit = vit_b_16.encoder
        self.pos_embedding = nn.Parameter(
            torch.empty(1, self.num_frames, 768).normal_(std=0.02)
        )

        face_layer_list = []
        for i in range(self.num_encoders):
            if self.adapter:
                if self.adapter_type == "nlp":
                    face_layer_list.append(
                        vit_adapter_nlp(transformer_encoder=vit.layers[i])
                    )
                else:
                    face_layer_list.append(
                        vit_adapter_conv(transformer_encoder=vit.layers[i])
                    )
            else:
                for p in vit.layers[i].parameters():
                    p.requires_grad = True
                face_layer_list.append(vit.layers[i])

        self.cnn_feature_extractor = cnn_face()
        self.VISUAL_TRANSFORMER = nn.Sequential(*face_layer_list)

        # ========== TEXT MODEL ==========
        bert = BertModel.from_pretrained("bert-base-uncased")
        for p in bert.parameters():
            p.requires_grad = False

        self.TEXT_EMBEDDINGS = bert.embeddings

        text_layer_list = []
        for i in range(self.num_encoders):
            if self.adapter:
                if self.adapter_type == "nlp":
                    text_layer_list.append(bert_adapter_nlp(bert.encoder.layer[i]))
                else:
                    text_layer_list.append(bert_adapter_conv(bert.encoder.layer[i]))
            else:
                for p in bert.encoder.layer[i].parameters():
                    p.requires_grad = True
                text_layer_list.append(bert.encoder.layer[i])

        self.TEXT_TRANSFORMER = nn.Sequential(*text_layer_list)

        # ========== FUSION & CLASSIFIER ==========
        if self.fusion_type == "concat":
            self.classifier = nn.Sequential(nn.Linear(768 * 3, 2))
        elif self.fusion_type == "cross2":
            self.cross_av = nn.Sequential(
                *[CrossFusionModule(dim=256) for _ in range(self.num_encoders)]
            )
            self.cross_at = nn.Sequential(
                *[CrossFusionModule(dim=256) for _ in range(self.num_encoders)]
            )
            self.cross_vt = nn.Sequential(
                *[CrossFusionModule(dim=256) for _ in range(self.num_encoders)]
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * self.num_encoders * 3, 64),
                nn.Dropout(p=0.5),
                nn.Linear(64, 2),
            )
            if self.multi:
                self.audio_classifier = nn.Linear(768, 2)
                self.vision_classifier = nn.Linear(768, 2)
                self.text_classifier = nn.Linear(768, 2)
        else:
            self.classifier = nn.Sequential(nn.Linear(768 * 3, 2))

    def forward(self, audio_input, visual_input, input_ids, attention_mask):
        # Audio processing
        audio_feat, _ = self.FEATURE_EXTRACTOR(audio_input, None)
        audios = self.FEATURE_PROJECTOR(audio_feat)

        # Visual processing
        b_s, no_of_frames, C, H, W = visual_input.shape
        visual_reshaped = torch.reshape(visual_input, (b_s * no_of_frames, C, H, W))
        faces = self.cnn_feature_extractor(visual_reshaped)
        faces = torch.reshape(faces, (b_s, no_of_frames, 256))
        faces = self.projection(faces) + self.pos_embedding

        # Text processing
        text_emb = self.TEXT_EMBEDDINGS(input_ids)
        texts = text_emb

        feat_ls = []
        if self.fusion_type == "concat":
            audios = self.AUDIO_TRANSFORMER(audios)
            faces = self.VISUAL_TRANSFORMER(faces)
            texts = self.TEXT_TRANSFORMER(texts)
            
            seq_len = min(audios.size(1), faces.size(1), texts.size(1))
            audios_pooled = torch.mean(audios[:, :seq_len, :], dim=1)
            faces_pooled = torch.mean(faces[:, :seq_len, :], dim=1)
            texts_pooled = torch.mean(texts[:, :seq_len, :], dim=1)
            
            fused_output = torch.cat((audios_pooled, faces_pooled, texts_pooled), dim=-1)
            
        elif self.fusion_type == "cross2":
            assert len(self.AUDIO_TRANSFORMER) == len(
                self.VISUAL_TRANSFORMER
            ), "unmatched encoders between audio and visual"
            assert len(self.AUDIO_TRANSFORMER) == len(
                self.TEXT_TRANSFORMER
            ), "unmatched encoders between audio and text"

            for i in range(self.num_encoders):
                audios = self.AUDIO_TRANSFORMER[i](audios)
                faces = self.VISUAL_TRANSFORMER[i](faces)
                texts = self.TEXT_TRANSFORMER[i](texts)

                seq_len = min(audios.size(1), faces.size(1), texts.size(1))
                fused_av = self.cross_av[i](audios[:, :seq_len, :], faces[:, :seq_len, :])
                fused_at = self.cross_at[i](audios[:, :seq_len, :], texts[:, :seq_len, :])
                fused_vt = self.cross_vt[i](faces[:, :seq_len, :], texts[:, :seq_len, :])

                feat_ls.extend([fused_av, fused_at, fused_vt])
            fused_output = torch.cat(feat_ls, dim=-1)
        else:
            raise Exception("undefined fusion type")

        logits = self.classifier(fused_output)

        if self.multi:
            a_logits = self.audio_classifier(torch.mean(audios, dim=1))
            v_logits = self.vision_classifier(torch.mean(faces, dim=1))
            t_logits = self.text_classifier(torch.mean(texts, dim=1))
            return (
                torch.mean(logits, 1) if logits.dim() == 3 else logits,
                a_logits,
                v_logits,
                t_logits,
            )
        else:
            return torch.mean(logits, 1) if logits.dim() == 3 else logits, None, None, None


if __name__ == "__main__":
    # for testing trimodal model:
    model = Fusion(
        "concat", num_encoders=2, adapter=True, adapter_type="nlp"
    )
    print(model)
    
    audio_input = torch.rand(8, 20601)
    visual_input = torch.rand(8, 64, 3, 160, 160)
    input_ids = torch.randint(0, 30522, (8, 64))
    attention_mask = torch.ones(8, 64)
    
    out, _, _, _ = model(audio_input, visual_input, input_ids, attention_mask)
    print(f"Output shape: {out.shape}")


"""
fusion types:

1- concat: simple concatenation of pooled outputs from audio, visual, and text models
2- cross2: cross-attention between all modality pairs (audio-visual, audio-text, visual-text) at each encoder layer
"""
