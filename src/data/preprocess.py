import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standard ImageNet normalization values for pre-trained vision models
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

class TextPreprocessor:
    def __init__(self, model_name='distilbert-base-uncased', max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def preprocess(self, text: str):
        encoded = self.tokenizer(
            str(text),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

class ImagePreprocessor:
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

    def preprocess(self, image):
        """
        image: PIL Image
        Returns preprocessed torch tensor of shape (3, image_size, image_size).
        If image is already a tensor, return with proper resizing/normalization.
        """
        return self.transform(image)

class MetadataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, metadata_matrix):
        """Fit the scaler on the training metadata."""
        self.scaler.fit(metadata_matrix)
        self.is_fitted = True

    def preprocess(self, metadata_vector):
        """Apply scaling to a single sample or batch."""
        if not self.is_fitted:
            # Fallback if not fitted
            return torch.tensor(metadata_vector, dtype=torch.float)
        
        # Ensure correct shape
        vector = np.array(metadata_vector).reshape(1, -1) if np.ndim(metadata_vector) == 1 else metadata_vector
        scaled = self.scaler.transform(vector)
        return torch.tensor(scaled, dtype=torch.float).squeeze(0)
