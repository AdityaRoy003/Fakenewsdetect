import torch
import torch.nn as nn
from .extractors import TextExtractor, ImageExtractor, MetadataExtractor

class MultimodalFusionModel(nn.Module):
    def __init__(self, metadata_input_dim, text_dim=768, image_dim=768, meta_dim=128, num_classes=1):
        super().__init__()
        self.text_extractor = TextExtractor(hidden_dim=text_dim)
        self.image_extractor = ImageExtractor(hidden_dim=image_dim)
        self.meta_extractor = MetadataExtractor(input_dim=metadata_input_dim, hidden_dim=meta_dim)
        
        fusion_dim = text_dim + image_dim + meta_dim
        
        # Self-attention module for the fused features
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, text_ids, text_mask, image, metadata):
        t_feat = self.text_extractor(text_ids, text_mask)
        i_feat = self.image_extractor(image)
        m_feat = self.meta_extractor(metadata)
        
        # Combine explicitly
        combined = torch.cat([t_feat, i_feat, m_feat], dim=1)
        
        # Apply attention gating
        attn_weights = self.attention(combined)
        attended_features = combined * attn_weights
        
        # Final classification
        out = self.classifier(attended_features)
        return out
