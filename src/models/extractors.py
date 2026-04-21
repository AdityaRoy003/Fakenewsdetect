import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel

class TextExtractor(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', hidden_dim=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        # DistilBERT doesn't have a pooler, we take the CLS token representation
        pooled_output = hidden_state[:, 0]
        return self.fc(pooled_output)

class ImageExtractor(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        # Load a pretrained ResNet18 and chop off the last fully connected layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, hidden_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1).squeeze(-1)  # remove spatial dimensions
        return self.fc(x)

class MetadataExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)
