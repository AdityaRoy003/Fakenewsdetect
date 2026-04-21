import pytest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.fusion import MultimodalFusionModel

def test_model_forward():
    # Meta input dim = 2 (e.g. author_credibility, domain_reputation)
    model = MultimodalFusionModel(metadata_input_dim=2, text_dim=768, image_dim=768, num_classes=1)
    model.eval()

    # Dummy inputs (batch size = 2)
    text_ids = torch.randint(0, 30000, (2, 32)) 
    text_mask = torch.ones((2, 32))
    image = torch.randn(2, 3, 224, 224)
    metadata = torch.randn(2, 2)

    with torch.no_grad():
        output = model(text_ids, text_mask, image, metadata)

    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
