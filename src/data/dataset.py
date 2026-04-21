import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class MultimodalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, text_preprocessor, image_preprocessor, metadata_preprocessor):
        """
        Args:
            df: Pandas DataFrame containing ['text', 'image_filename', 'label'] and additional metadata columns.
            image_dir: Base directory for image files.
        """
        self.df = df
        self.image_dir = image_dir
        self.text_preprocessor = text_preprocessor
        self.image_preprocessor = image_preprocessor
        self.metadata_preprocessor = metadata_preprocessor
        
        self.meta_cols = [col for col in df.columns if col not in ['text', 'image_filename', 'label']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text
        text = str(row['text']) if 'text' in row else ""
        text_features = self.text_preprocessor.preprocess(text)
        
        # Image
        image_features = torch.zeros((3, 224, 224))
        if 'image_filename' in row and pd.notna(row['image_filename']):
            image_path = os.path.join(self.image_dir, str(row['image_filename']))
            try:
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    image_features = self.image_preprocessor.preprocess(image)
            except Exception:
                pass # Use zero tensor fallback
                
        # Metadata
        if len(self.meta_cols) > 0:
            meta_vector = row[self.meta_cols].values
        else:
            meta_vector = [0.0]  # Fallback dummy metadata
        meta_features = self.metadata_preprocessor.preprocess(meta_vector)
        
        # Label
        label = torch.tensor(row['label'], dtype=torch.float) if 'label' in row else torch.tensor(-1.0)
        
        return {
            'text_input_ids': text_features['input_ids'],
            'text_attention_mask': text_features['attention_mask'],
            'image': image_features,
            'metadata': meta_features,
            'label': label
        }
