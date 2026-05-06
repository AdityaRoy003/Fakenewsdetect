import pandas as pd
import numpy as np
import os
from PIL import Image

def prepare_kaggle_dataset(fake_csv_path, real_csv_path, output_csv_path, images_dir="data/images"):
    print("Loading Kaggle datasets...")
    # Load data
    try:
        fake_df = pd.read_csv(fake_csv_path)
        real_df = pd.read_csv(real_csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {fake_csv_path} or {real_csv_path}")
        print("Please download the 'Fake and real news dataset' from Kaggle and place the CSVs in the data folder.")
        return

    # Add labels (1 for Fake, 0 for Real)
    fake_df['label'] = 1
    real_df['label'] = 0

    # Combine datasets
    df = pd.concat([fake_df, real_df], ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # We only need the 'text' column for the core analysis, but the Kaggle dataset also has 'title'
    # Let's combine title and text for maximum context
    df['text'] = df['title'] + " " + df['text']
    
    print(f"Combined dataset size: {len(df)} articles.")
    
    # --- Generate Missing Multimodal Data ---
    print("Generating simulated metadata and images for multimodal pipeline...")
    
    # 1. Author Credibility & Domain Reputation
    # Real news tends to have higher credibility, Fake news lower, but we add noise for realism
    df['author_credibility'] = np.where(df['label'] == 0, np.random.uniform(0.6, 1.0, len(df)), np.random.uniform(0.1, 0.5, len(df)))
    df['domain_reputation'] = np.where(df['label'] == 0, np.random.uniform(0.7, 1.0, len(df)), np.random.uniform(0.0, 0.4, len(df)))

    # 2. Images
    # Since ISOT is a text-only dataset, our MultimodalFusionModel expects images.
    # We will generate a single dummy blank image and point all articles to it to satisfy the architecture.
    os.makedirs(images_dir, exist_ok=True)
    dummy_img_path = os.path.join(images_dir, "dummy_image.jpg")
    
    if not os.path.exists(dummy_img_path):
        img = Image.new('RGB', (224, 224), color = (73, 109, 137))
        img.save(dummy_img_path)
        
    df['image_filename'] = "dummy_image.jpg"
    
    # Keep only the columns expected by train.py
    final_df = df[['text', 'image_filename', 'author_credibility', 'domain_reputation', 'label']]
    
    # Save a smaller subset for faster testing (e.g., 5000 rows instead of 40k)
    subset_df = final_df.head(5000)
    subset_df.to_csv(output_csv_path, index=False)
    
    print(f"Successfully saved prepared dataset to {output_csv_path}!")
    print("You can now modify scripts/train.py to use this CSV and begin training your AI!")

if __name__ == "__main__":
    # Assuming the user places Fake.csv and True.csv in the root folder
    prepare_kaggle_dataset(
        fake_csv_path="Fake.csv", 
        real_csv_path="True.csv", 
        output_csv_path="prepared_dataset.csv"
    )
