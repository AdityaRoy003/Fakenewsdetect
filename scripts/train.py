import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os

# Ensure src in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import MultimodalDataset
from src.data.preprocess import TextPreprocessor, ImagePreprocessor, MetadataPreprocessor
from src.models.fusion import MultimodalFusionModel
from src.training.lightning_module import FakeNewsLightningModule

def main():
    # Setup dummy data to verify the pipeline
    print("Initializing dummy dataset...")
    df = pd.DataFrame({
        'text': ["Real news article text here.", "Fake news article text here."],
        'image_filename': ["real.jpg", "fake.jpg"],
        'author_credibility': [0.95, 0.12],
        'domain_reputation': [0.88, 0.20],
        'label': [0, 1]  # 0 for real, 1 for fake
    })
    
    # Initialize Preprocessors
    print("Initializing preprocessors...")
    text_prep = TextPreprocessor(max_length=32)
    img_prep = ImagePreprocessor(image_size=128)
    meta_prep = MetadataPreprocessor()
    
    # Fit metadata preprocessor
    meta_features = df[['author_credibility', 'domain_reputation']].values
    meta_prep.fit(meta_features)

    # Initialize Dataset
    dataset = MultimodalDataset(df, "data/images", text_prep, img_prep, meta_prep)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize Model
    print("Initializing model...")
    # Meta input dim = 2 (author_credibility, domain_reputation)
    model = MultimodalFusionModel(metadata_input_dim=2, text_dim=768, image_dim=768, num_classes=1)
    
    # Initialize Lightning Module
    lightning_module = FakeNewsLightningModule(model, lr=1e-4)
    
    # Train
    print("Starting training...")
    trainer = pl.Trainer(max_epochs=2, accelerator="auto")
    trainer.fit(lightning_module, train_dataloaders=dataloader, val_dataloaders=dataloader)

if __name__ == "__main__":
    main()
