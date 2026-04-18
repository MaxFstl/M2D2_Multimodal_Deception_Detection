import torch
import torch.nn as nn
from transformers import BertModel
from src.models.adapter import bert_adapter_nlp, bert_adapter_conv


class TextModel(nn.Module):
    def __init__(self, num_encoders: int, adapter: bool, adapter_type: str):
        super(TextModel, self).__init__()

        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type

        bert = BertModel.from_pretrained("bert-base-uncased")
        for p in bert.parameters():
            p.requires_grad = False

        self.EMBEDDINGS = bert.embeddings

        layer_list = []
        for i in range(self.num_encoders):
            if self.adapter:
                if self.adapter_type == "nlp":
                    layer_list.append(bert_adapter_nlp(bert.encoder.layer[i]))
                else:
                    layer_list.append(bert_adapter_conv(bert.encoder.layer[i]))
            else:
                for p in bert.encoder.layer[i].parameters():
                    p.requires_grad = True
                layer_list.append(bert.encoder.layer[i])

        self.TRANSFORMER = nn.Sequential(*layer_list)

        self.classifier = nn.Sequential(nn.Linear(768, 2))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.EMBEDDINGS(input_ids)
        output_tokens = self.TRANSFORMER(embeddings)
        logits = self.classifier(output_tokens)
        return torch.mean(logits, 1)
